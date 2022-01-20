# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import six
import tensorflow as tf

from utils import get_shape_list
from utils.my_attention import my_self_relative_attention_v2
from utils.positional import add_timing_signal_1d


# from tensor2tensor.layers.common_attention import dot_product_attention_relative, dot_product_unmasked_self_attention_relative_v2
# from sparse_attention.attention import blocksparse_attention_impl


class BertModel(object):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 input_ids=None,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 scope=None,
                 encoder_output=None,
                 encoder_mask=None,
                 trainable_layers=0,
                 embedding_trainable=False,
                 pooler_layer_trainable=True,
                 masked_layer_trainable=True,
                 attention_layer_trainable=True):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. rue for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids(DEPRECATED): int32 Tensor of shape [batch_size, seq_length].
            input_mask(DEPRECATED): (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
                it is must faster if this is True, on the CPU or GPU, it is faster if
                this is False.
            scope: (optional) variable scope. Defaults to "bert".
            # memory_tensor: Memory for transformer block to calc attention(the output of Encoder part).

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        # config.hidden_dropout_prob = 0.0
        # config.attention_probs_dropout_prob = 0.0

        # input_shape = get_shape_list(input_ids, expected_rank=2)
        shape_list = get_shape_list(input_ids)
        batch_size = shape_list[0]
        seq_length = shape_list[1]

        # add by tangb
        # self.input = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
        # self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
        # self.memory_tensor = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, config.hidden_size])
        # input_ids = self.input
        # input_mask = self.input_mask
        # finish

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        token_type_flag = True
        self.topk = None

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
            token_type_flag = False

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            # Perform embedding lookup on the word ids.
            (self.embedding_output, self.embedding_table) = embedding_lookup(
                input_ids=input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings,
                trainable=embedding_trainable)

            # Add positional embeddings and token type embeddings, then layer
            # normalize and perform dropout.
            self.embedding_output = embedding_postprocessor(
                input_tensor=self.embedding_output,
                use_token_type=token_type_flag,
                token_type_ids=token_type_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=hidden_dropout_prob,
                trainable=embedding_trainable)

            # Add start embedding
            # self.embedding_output: shape=(batch_size, seq_length, hidden_dim(768))

            if encoder_output is not None:
                with tf.variable_scope('shift'):
                    start_embedding = encoder_output[:, 0, :] if config.align_layers else encoder_output[:, 0, :]
                    start_embedding = tf.expand_dims(start_embedding, axis=1)
                    self.embedding_output = tf.concat([start_embedding, self.embedding_output], axis=1)[:, :-1, :]
                    # input_mask = tf.concat([tf.ones(shape=[batch_size, 1], dtype=tf.int32),
                    #                         input_mask], axis=1)[:, :-1]
                    # input_ids = tf.concat([tf.ones(shape=[batch_size, 1], dtypte=tf.int32) * 102, input_ids], axis=1)[:, :-1]

        with tf.variable_scope(scope, default_name="bert", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("encoder"):

                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.

                # this one is used for raw attention.
                # attention_mask = create_attention_mask_from_input_mask(input_mask, input_mask)
                # this one is used for my_attention.

                if encoder_output is not None and encoder_mask is not None:
                    # generate masked self attention mask.
                    decoder_mask = tf.constant(np.tril(np.ones(shape=(seq_length, seq_length))), name='decoder_mask')
                    decoder_mask = tf.tile(tf.expand_dims(decoder_mask, 0), [batch_size, 1, 1])
                    padding_mask = create_attention_mask_from_input_mask(input_mask, input_mask)
                    self_attention_mask = tf.cast(tf.cast(decoder_mask, tf.bool) & tf.cast(padding_mask, tf.bool),
                                                  tf.float32)
                    self.self_attention_mask = self_attention_mask

                    # generate encoder-decoder attention mask.
                    enc_dec_attention_mask = create_attention_mask_from_input_mask(
                        input_mask, encoder_mask[:, :])
                else:
                    # this one is used for my_attention_v2
                    self_attention_mask = input_mask
                    # this one is used for formal attention
                    # self_attention_mask = create_attention_mask_from_input_mask(input_mask, input_mask)
                    self.self_attention_mask = self_attention_mask
                    enc_dec_attention_mask = None

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers, self.all_attention_probs, self.all_self_attention_probs = transformer_model(
                    config=config,
                    input_tensor=self.embedding_output,
                    self_attention_mask=self_attention_mask,
                    enc_dec_attention_mask=enc_dec_attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True,
                    memory_tensor=encoder_output,
                    trainable_layers=trainable_layers,
                    masked_layer_trainable=masked_layer_trainable,
                    attention_layer_trainable=attention_layer_trainable,
                    do_return_attention_probs=True,
                    use_trim_attention=config.trim_attention,
                )

            self.sequence_output = self.all_encoder_layers[-1]
            self.sequence_mask = input_mask

            self.encoder_output_before = self.sequence_output


            # get top n segment infomation from sequence_output
            if encoder_output is None and \
                    config.encoder_seq_length != config.encoder_output_length:

                #  -------get different parts embedding------
                # get [SENTEN] embedding
                # senten_embedding: [batch size, hidden size]
                senten_embedding = self.sequence_output[:, 0, :]

                # get [SEP] embedding
                # sep_embedding: [batch size, hidden size]
                # sep_embedding = self.sequence_output[:, -1, :]

                # [CLS] token position : [1, 17, ......]
                cls_positions = []
                for i in range(0, config.encoder_seq_length - 1, config.segment_length + 1):
                    cls_positions.append(i + 1)
                cls_list = [self.sequence_output[:, cls_id, :] for cls_id in cls_positions]
                cls_embedding = tf.concat(cls_list, 1)
                # get [CLS] embedding
                # cls_embedding: [batch size, segment number, hidden size]
                cls_embedding = tf.reshape(cls_embedding, [-1, config.segment_number, config.hidden_size])

                # -------calculate similarity-------
                if config.similarity_function[0] == "cos":
                    cls_attention, distance = cos_distance(
                        tf.reshape(senten_embedding, [-1, 1, config.hidden_size]), cls_embedding)
                elif config.similarity_function[0] == "euclidean":
                    cls_attention, distance = euclidean_distance(
                        tf.reshape(senten_embedding, [-1, 1, config.hidden_size]), cls_embedding)
                elif config.similarity_function[0] == "manhattan":
                    cls_attention, distance = manhattan_distance(
                        tf.reshape(senten_embedding, [-1, 1, config.hidden_size]), cls_embedding)
                elif config.similarity_function[0] == "mahalanobis":
                    cls_attention, distance = mahalanobis_distance(
                        tf.reshape(senten_embedding, [-1, 1, config.hidden_size]), cls_embedding)
                else:
                    print("Invalid similarity function! Please choose from {}, {}, {}, {}.".format(
                        "cos", "euclidean", "manhattan", "mahalanobis"))
                    exit(1)

                # cls_attention: [batch size, segment number]
                cls_attention = tf.reshape(cls_attention, [-1, config.segment_number])
                # cls_attention: [batch size, segment number]
                cls_attention = tf.nn.softmax(cls_attention)



                # ------- make segment mask -------
                # segment mask with segment id : [0, 0, 1, ... , 1, 0, 2, ..., 2, ......., 15, 0]
                # segment_mask = []
                # seg_cnt = 0
                # for i in range(config.encoder_seq_length):
                #     if (i - 1) % (config.segment_length + 1) == 0:
                #         seg_cnt += 1
                #     segment_mask.append(seg_cnt)

                # segment mask with segment id : [0, 0, 1, ... , 1, 0, 2, ..., 2, ......., 15, 0]
                segment_mask = []
                seg_cnt = 0
                for i in range(config.encoder_seq_length):
                    if (i - 1) % (config.segment_length + 1) == 0:
                        seg_cnt += 1
                        segment_mask.append(seg_cnt)
                    else:
                        segment_mask.append(0)


                # segment_mask[-1] = 0  # [SEP] -> 0
                segment_mask = tf.constant(np.array(segment_mask))
                segment_mask = tf.reshape(segment_mask, [1, 1, config.encoder_seq_length])
                # segment_mask: [batch_size, k, seq length]
                segment_mask = tf.tile(segment_mask, [batch_size, config.top_k, 1])

                # ------- get top k segment --------
                # top_k_index: [batch_size, k]
                _, top_k_index = tf.nn.top_k(cls_attention, config.top_k)
                self.topk = top_k_index
                top_k_index = tf.reshape(top_k_index, [batch_size, config.top_k, 1]) + 1
                # top_k_index: [batch_size, k, seq length]
                top_k_index = tf.tile(top_k_index, [1, 1, config.encoder_seq_length])

                # int32 -> int64
                segment_mask = tf.cast(segment_mask, dtype=tf.int64)
                top_k_index = tf.cast(top_k_index, dtype=tf.int64)
                # segment_mask: [batch_size, k, seq length]
                segment_mask = tf.equal(segment_mask, top_k_index)
                segment_mask = tf.cast(segment_mask, dtype=tf.int64)

                # segment_mask: [batch_size, seq length]
                segment_mask = tf.reduce_sum(segment_mask, 1)
                segment_mask_non_be = segment_mask[:, 1:]
                one_mask = tf.ones([batch_size, 1], dtype=tf.int64)

                # segment_mask_non_be: [batch_size, seq length]
                segment_mask_non_be = tf.concat([one_mask, segment_mask_non_be], 1)
                # segment_mask: [batch_size, seq length]
                segment_mask = tf.cast(segment_mask, dtype=tf.bool)

                tmp_output = tf.boolean_mask(self.sequence_output, segment_mask)
                # tmp_output: [batch_size, (seg len+1) x k, hidden size]
                # tmp_output = tf.reshape(tmp_output,
                #                         [batch_size, config.top_k * (config.segment_length + 1), config.hidden_size])

                tmp_output = tf.reshape(tmp_output,
                                        [batch_size, config.top_k, config.hidden_size])

                senten_embedding = tf.reshape(senten_embedding, [batch_size, 1, config.hidden_size])
                # sep_embedding = tf.reshape(sep_embedding, [batch_size, 1, config.hidden_size])
                # tmp_output: [batch_size, (seg len+1) x k + 2, hidden size]
                # tmp_output = tf.concat([senten_embedding, tmp_output, sep_embedding], 1)
                tmp_output = tf.concat([senten_embedding, tmp_output], 1)

                # sequence_output: [batch_size, (seg len+1) x k + 1, hidden size]
                self.sequence_output = tmp_output
                self.encoder_output_after = tmp_output
                self.sequence_mask = tf.boolean_mask(input_mask, segment_mask_non_be)
                # sequence_mask: [batch size, (seg len+1) x k + 1]
                # self.sequence_mask = tf.reshape(self.sequence_mask,
                #                                 [batch_size, config.top_k * (config.segment_length + 1) + 1])
                self.sequence_mask = tf.reshape(self.sequence_mask,
                                                [batch_size, config.top_k + 1])

                # self.distance = distance  # todo:get distance
                # print("!!!!!!!!!sequence_mask:", get_shape_list(self.sequence_mask))

            # self.attention_prob = self.all_attention_probs[-1]
            # by tangbin: we don't need this pooler operation.
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            # with tf.variable_scope("pooler"):
            #     # We "pool" the model by simply taking the hidden state corresponding
            #     # to the first token. We assume that this has been pre-trained
            #     first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
            #     self.pooled_output = tf.layers.dense(
            #         first_token_tensor,
            #         config.hidden_size,
            #         activation=tf.tanh,
            #         kernel_initializer=create_initializer(config.initializer_range),
            #         trainable=pooler_layer_trainable)

    # def get_pooled_output(self):
    #     return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_sequence_mask(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_mask

    def get_distance(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.distance

    def get_topk(self):
        return self.topk

    def get_encoder_output_before(self):
        return self.encoder_output_before

    def get_encoder_output_after(self):
        return self.encoder_output_after

    def get_embedding(self):
        return self.embedding_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None, trainable=True):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name, trainable=trainable)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None, trainable=True):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name, trainable=trainable)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False,
                     trainable=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
        for TPUs.
      trainable: trainable or not

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
        trainable=trainable)

    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output, embedding_table


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            trainable=False):
    """Performs various post-processing on a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range),
            trainable=trainable)
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])

        # one_hot_ids: [sequence length - 2, segment number]
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)


        # [senten] and [SEP] position
        # token embedding set to zeros
        # senten_seq_token_type_embedding: [1, segment number]
        senten_seq_token_type_embedding = tf.zeros([1, token_type_vocab_size], dtype=tf.float32, name=None)
        # one_hot_ids: [sequence length, segment number]
        one_hot_ids = tf.concat([senten_seq_token_type_embedding, one_hot_ids],
                                0)
        # # one_hot_ids: [1, sequence length, segment number]
        # one_hot_ids = tf.reshape(one_hot_ids, [1, seq_length, token_type_vocab_size])

        # # one_hot_ids: [batch_size, sequence length, segment number]
        # one_hot_ids = tf.tile(one_hot_ids, [batch_size, 1, 1])

        # token_type_embeddings: [sequence length, hidden size]
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)

        token_type_embeddings = tf.reshape(token_type_embeddings, [1, seq_length, width])
        token_type_embeddings = tf.tile(token_type_embeddings, [batch_size, 1, 1])

        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        output = add_timing_signal_1d(output,
                                      min_timescale=1.0,
                                      max_timescale=5.0e3)
        # INFO: the code below is what BERT use.
        # assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        # with tf.control_dependencies([assert_op]):
        #     full_position_embeddings = tf.get_variable(
        #         name=position_embedding_name,
        #         shape=[max_position_embeddings, width],
        #         initializer=create_initializer(initializer_range),
        #         trainable=trainable)
        #     # Since the position embedding table is a learned variable, we create it
        #     # using a (long) sequence length `max_position_embeddings`. The actual
        #     # sequence length might be shorter than this, for faster training of
        #     # tasks that do not have long sequences.
        #     #
        #     # So `full_position_embeddings` is effectively an embedding table
        #     # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        #     # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        #     # perform a slice.
        #     position_embeddings = tf.slice(full_position_embeddings, [0, 0],
        #                                    [seq_length, -1])
        #     num_dims = len(output.shape.as_list())
        #
        #     # Only the last two dimensions are relevant (`seq_length` and `width`), so
        #     # we broadcast among the first dimensions, which is typically just
        #     # the batch size.
        #     position_broadcast_shape = []
        #     for _ in range(num_dims - 2):
        #         position_broadcast_shape.append(1)
        #     position_broadcast_shape.extend([seq_length, width])
        #     position_embeddings = tf.reshape(position_embeddings,
        #                                      position_broadcast_shape)
        #     output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob, trainable=trainable)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    # broadcast_ones = tf.ones(
    #     shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    from_mask = tf.cast(tf.reshape(from_tensor, [batch_size, from_seq_length, 1]), dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = from_mask * to_mask

    return mask


def my_attention_layer(config,
                       from_tensor,
                       to_tensor,
                       attention_mask=None,
                       num_attention_heads=1,
                       size_per_head=512,
                       query_act=None,
                       key_act=None,
                       value_act=None,
                       attention_probs_dropout_prob=0.0,
                       initializer_range=0.02,
                       do_return_2d_tensor=False,
                       batch_size=None,
                       from_seq_length=None,
                       to_seq_length=None,
                       trainable=True,
                       masked=True):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        # shape of input_tensor: [batch_size * seq_length, num_heads * size_per_head]
        # shape of output_tensor: [batch_size, seq_length, num_heads* size_per_head]
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads * width])
        # shape of output_tensor: [batch_size, num_heads, seq_length, size_per_head] / [batch, heads, length, depth].
        # output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    # delete by tangb
    # if len(from_shape) != len(to_shape):
    # raise ValueError(
    #     "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # TODO：添加segment mask(encoder 的self attention 部分添加attention mask)
    # 判断是否是是encoder self attention的部分？
    segment_mask = None
    need_segment_mask = False
    if from_seq_length == to_seq_length and from_seq_length == config.encoder_seq_length:
        need_segment_mask = True
    if need_segment_mask:
        segment_number = config.segment_number
        segment_length = config.segment_length

        # segment_mask = np.ones((from_seq_length, from_seq_length))
        # for i in range(segment_number):
        #     segment_start = i * (segment_length + 1) + 1
        #     segment_end = (i + 1) * (segment_length + 1)
        #     #     print(segment_start, segment_end)
        #
        #     for j in range(1, from_seq_length):
        #         if j >= segment_start and j <= segment_end:
        #             continue
        #         segment_mask[segment_start][j] = 0
        #         segment_mask[j][segment_start] = 0

        # segment_mask = np.zeros((from_seq_length, from_seq_length))
        # for i in range(segment_number):
        #     segment_start = i * (segment_length + 1) + 1
        #     segment_end = (i + 1) * (segment_length + 1)
        #     for p in range(segment_start, segment_end+1):
        #         for q in range(segment_start, segment_end + 1):
        #             segment_mask[p][q] = 1
        # for i in range(from_seq_length):
        #     segment_mask[0][i] = 1
        #     segment_mask[i][0] = 1
        #
        # segment_mask = tf.constant(segment_mask)
        # segment_mask = tf.cast(segment_mask, tf.float32)
        #
        # 添加mask
        # from_tensor = from_tensor * segment_mask
        # to_tensor = to_tensor * segment_mask

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range),
        trainable=trainable)

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range),
        trainable=trainable)

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range),
        trainable=trainable)

    # `query_layer` = [B, N, F*H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T*H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # `value_layer` = [B, N, T*H]
    value_layer = transpose_for_scores(value_layer, batch_size,
                                       num_attention_heads, to_seq_length,
                                       size_per_head)

    # The following steps are calculating attention between Q, K and V, so we modify it.

    # context_layer: [batch_size, seq_length, num_heads*size_per_head] / [B, N, F, H]
    # sparse attention:
    # context_layer = blocksparse_attention_impl(query_layer,
    #                                            key_layer,
    #                                            value_layer,
    #                                            heads=num_attention_heads,
    #                                            attn_mode="fixed",
    #                                            local_attn_ctx=128,
    #                                            num_verts=4,
    #                                            vertsize=1,
    #                                            recompute=True,
    #                                            masked=masked)
    # my self relative attention:
    context_layer = my_self_relative_attention_v2(q=query_layer,
                                                  k=key_layer,
                                                  v=value_layer,
                                                  mask_matrix=attention_mask,
                                                  relative_size=8,
                                                  heads=num_attention_heads)
    # # `context_layer` = [B, F, N, H]
    # context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    #
    # if do_return_2d_tensor:
    #     # `context_layer` = [B*F, N*V]
    #     context_layer = tf.reshape(
    #         context_layer,
    #         [batch_size * from_seq_length, num_attention_heads * size_per_head])
    # else:
    #     # `context_layer` = [B, F, N*V]
    #     context_layer = tf.reshape(
    #         context_layer,
    #         [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def attention_layer(config,
                    from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    trainable=True,
                    return_attention_probs=False,
                    **kwargs):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    # delete by tangb
    # if len(from_shape) != len(to_shape):
    # raise ValueError(
    #     "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # TODO：添加segment mask(encoder 的self attention 部分添加attention mask)
    # 判断是否是是encoder self attention的部分？
    # segment_mask = None
    # need_segment_mask = False
    # if from_seq_length == to_seq_length and from_seq_length == config.encoder_seq_length:
    #     need_segment_mask = True
    # if need_segment_mask:
    #
    #     segment_number = config.segment_number
    #     segment_length = config.segment_length


        # segment_mask = np.ones((from_seq_length, from_seq_length))
        # for i in range(segment_number):
        #     segment_start = i * (segment_length + 1) + 1
        #     segment_end = (i + 1) * (segment_length + 1)
        #     #     print(segment_start, segment_end)
        #
        #     for j in range(1, from_seq_length):
        #         if j >= segment_start and j <= segment_end:
        #             continue
        #         segment_mask[segment_start][j] = 0
        #         segment_mask[j][segment_start] = 0

        # segment_mask = np.zeros((from_seq_length, from_seq_length))
        # for i in range(segment_number):
        #     segment_start = i * (segment_length + 1) + 1
        #     segment_end = (i + 1) * (segment_length + 1)
        #     for p in range(segment_start, segment_end+1):
        #         for q in range(segment_start, segment_end + 1):
        #             segment_mask[p][q] = 1
        # for i in range(from_seq_length):
        #     segment_mask[0][i] = 1
        #     segment_mask[i][0] = 1
        #
        # segment_mask = tf.constant(segment_mask)
        # segment_mask = tf.cast(segment_mask, tf.float32)

        # 添加mask
        # from_tensor = from_tensor * segment_mask
        # to_tensor = to_tensor * segment_mask


    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range),
        trainable=trainable)

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range),
        trainable=trainable)

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range),
        trainable=trainable)

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)


    # if need_segment_mask:
    #     attention_scores = attention_scores * need_segment_mask
    #     # 添加segment mask

    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        if len(attention_mask.shape) == 2:
            # TODO
            # `attention_mask`: [B, F] -> [B, F, T]
            attention_mask = create_attention_mask_from_input_mask(attention_mask, attention_mask)
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.cast(tf.expand_dims(attention_mask, axis=[1]), tf.float32)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - attention_mask) * -1e8

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # adder = tf.Print(adder, data=[tf.shape(adder)], message='Adder Shape:', summarize=999999)
        # attention_scores = tf.Print(attention_scores, data=[tf.shape(attention_scores)], message='Attention Scores Shape:', summarize=99999)
        # print('attention score shape', attention_scores.shape)
        # print('adder shape', adder.shape)
        attention_scores = attention_scores + adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores) * attention_mask

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    if return_attention_probs:
        attention_probs_shaped = tf.transpose(attention_probs, [0, 2, 3, 1])
        probs = tf.reduce_mean(attention_probs_shaped, axis=-1)
        return context_layer, probs
    else:
        return context_layer


def transformer_model(config,
                      input_tensor,
                      self_attention_mask=None,
                      enc_dec_attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      memory_tensor=None,
                      trainable_layers=1,
                      masked_layer_trainable=True,
                      attention_layer_trainable=True,
                      do_return_attention_probs=True,
                      use_trim_attention=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.
      memory_tensor: float Tensor with the same shape of input_tensor. If it's None,
        the model is an Encoder(self-attention + feedforward net). If it's not None,
        the model is an Decoder(masked self-attention + attention + feedforward net).
      trainable_layers:

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))
    assert -1 <= trainable_layers <= num_hidden_layers

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)
    prev_output = input_tensor

    is_decoding = memory_tensor is not None
    if is_decoding:
        memory_tensor_shape_list = get_shape_list(memory_tensor[-1] if type(memory_tensor) == list else memory_tensor)
        # memory_tensor = reshape_to_matrix(memory_tensor)
        memory_tensor_seq_length = memory_tensor_shape_list[1]
    all_layer_outputs = []
    all_layer_attention_probs = []
    all_layer_self_attention_probs = []
    if trainable_layers == -1:
        trainables = [True] * num_hidden_layers
    else:
        trainables = [False] * (num_hidden_layers - trainable_layers) + [True] * trainable_layers
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            trainable = trainables[layer_idx]
            layer_input = prev_output

            # add by tangb: masked self-attention for decoder
            if is_decoding:
                with tf.variable_scope("masked_attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = attention_layer(
                            config=config,
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            attention_mask=self_attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=False,
                            batch_size=batch_size,
                            from_seq_length=config.encoder_output_length,  # TODO : fix from seq length
                            to_seq_length=config.encoder_output_length,  # TODO : fix to seq length
                            trainable=True if trainable else masked_layer_trainable,
                            masked=True)
                        attention_heads.append(attention_head)
                        # all_layer_self_attention_probs.append(self_attention_prob)

                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate
                        # them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis=-1)

                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=create_initializer(initializer_range),
                            trainable=trainable)
                        attention_output = dropout(attention_output, hidden_dropout_prob)
                        attention_output = layer_norm(attention_output + layer_input,
                                                      trainable=True if trainable else masked_layer_trainable)
                    layer_input = attention_output
            # end add.

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    func = attention_layer if not use_trim_attention or is_decoding else my_attention_layer
                    if is_decoding:
                        to_tensor = memory_tensor[layer_idx] if type(memory_tensor) == list else memory_tensor
                    else:
                        to_tensor = layer_input
                    attention_head, attention_prob = func(
                        config=config,
                        from_tensor=layer_input,
                        to_tensor=to_tensor,
                        attention_mask=enc_dec_attention_mask if is_decoding else self_attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=False,
                        batch_size=batch_size,
                        from_seq_length=config.encoder_output_length if is_decoding else seq_length,
                        # seq_length,  # TODO : fix seq len
                        to_seq_length=memory_tensor_seq_length if is_decoding else seq_length,  # TODO : fix seq len
                        trainable=True if trainable else attention_layer_trainable,
                        return_attention_probs=True,
                        masked=False)
                    attention_heads.append(attention_head)
                    all_layer_attention_probs.append(attention_prob)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range),
                        trainable=trainable)
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input,
                                                  trainable=True if trainable else attention_layer_trainable)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range),
                    trainable=trainable)

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range),
                    trainable=trainable)
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output, trainable=trainable)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        if do_return_attention_probs:
            return final_outputs, all_layer_attention_probs, all_layer_self_attention_probs
        else:
            return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        if do_return_attention_probs:
            return final_output, all_layer_attention_probs[-1], all_layer_self_attention_probs[-1]
        else:
            return final_output


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def cos_distance(X1, X2):
    """
    calculate cos distance
    big better
    :param X1: [batch size, x1 length, hidden size]
    :param X2: [batch size, x2 length, hidden size]
    :return:   [batch size, x1 length, x2 length]
    """
    X1_shape_list = get_shape_list(X1)
    X2_shape_list = get_shape_list(X2)

    assert len(X1_shape_list) == len(X2_shape_list)

    batch_size = X1_shape_list[0]
    X1_length = X1_shape_list[1]
    X2_length = X2_shape_list[1]

    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=-1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))

    X1_X2 = tf.matmul(X1, tf.transpose(X2, perm=[0, 2, 1]))
    X1_X2_norm = tf.matmul(tf.reshape(X1_norm, [batch_size, X1_length, 1]),
                           tf.reshape(X2_norm, [batch_size, 1, X2_length]))

    distance = X1_X2 / X1_X2_norm
    # distance = max_distance - distance

    return distance, distance


def euclidean_distance(X1, X2):
    """
    calculate euclidean distance
    big better (origin small better --> max_score - score)
    :param X1: [batch size, 1, hidden size]
    :param X2: [batch size, x2 length, hidden size]
    :return:   [batch size, x2 length]
    """

    X1_shape_list = get_shape_list(X1)
    X2_shape_list = get_shape_list(X2)

    assert len(X1_shape_list) == len(X2_shape_list)

    batch_size = X1_shape_list[0]
    X2_length = X2_shape_list[1]

    distance = tf.sqrt(tf.reduce_sum(tf.square(X2 - X1), -1))
    max_distance = tf.tile(tf.reshape(tf.reduce_max(distance, 1), [batch_size, 1]), [1, X2_length])
    # distance = max_distance - distance

    return max_distance - distance, distance


def manhattan_distance(X1, X2):
    """
    calculate manhattan distance
    big better (origin small better --> max_score - score)
    :param X1: [batch size, 1, hidden size]
    :param X2: [batch size, x2 length, hidden size]
    :return:   [batch size, x2 length]
    """

    X1_shape_list = get_shape_list(X1)
    X2_shape_list = get_shape_list(X2)

    assert len(X1_shape_list) == len(X2_shape_list)

    batch_size = X1_shape_list[0]
    X2_length = X2_shape_list[1]

    distance = tf.reduce_sum(X2 - X1, -1)
    max_distance = tf.tile(tf.reshape(tf.reduce_max(distance, 1), [batch_size, 1]), [1, X2_length])
    # distance = max_distance - distance

    return max_distance - distance, distance


def mahalanobis_distance(X1, X2):
    """
    calculate mahalanobis distance
    big better (origin small better --> max_score - score)
    :param X1: [batch size, 1, hidden size]
    :param X2: [batch size, x2 length, hidden size]
    :return:   [batch size, x2 length]
    """

    X1_shape_list = get_shape_list(X1)
    X2_shape_list = get_shape_list(X2)

    batch_size = X1_shape_list[0]
    X2_length = X2_shape_list[1]

    diff = X2 - X1
    conv = tf.matmul(tf.transpose(diff, perm=[0, 2, 1]), diff) / tf.cast(X2_length - 1, tf.float32)
    conv_inverse = tf.matrix_inverse(conv)

    distance = tf.matmul(diff, conv_inverse)
    distance = tf.matmul(distance, tf.transpose(diff, perm=[0, 2, 1]))
    distance = tf.sqrt(tf.reduce_sum(distance, -1))
    max_distance = tf.tile(tf.reshape(tf.reduce_max(distance, 1), [batch_size, 1]), [1, X2_length])
    # distance = max_distance - distance

    return max_distance - distance, distance
