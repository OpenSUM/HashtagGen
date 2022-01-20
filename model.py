import copy

import numpy as np
import tensorflow as tf

from constants import *
from modeling_bert import BertModel, create_initializer, attention_layer
from optimization import create_bert_optimizer, create_other_optimizer
from utils import eprint, get_shape_list


def get_initializer(name, **kwargs):
    if name.lower() == 'normal':
        return tf.random_normal_initializer()
    elif name.lower() == 'truncated':
        return tf.truncated_normal_initializer(stddev=kwargs['initializer_range'])
    elif name.lower() == 'xavier':
        return tf.glorot_normal_initializer()
    else:
        eprint('[WARNING: Activation function {} not found, use tf.random_normal_initializer() instead.'.format(name))
        return tf.random_normal_initializer()


class Pointer:
    def __init__(self, config, parent_model):
        self.parent_model = parent_model
        self.config = copy.deepcopy(config)

        self.attn_W_encoder = None
        self.attn_W_decoder = None
        self.attn_V = None

        self.W_pointer = None

        self.encoder_attn = None
        self.p_gen = None
        self.p_vocab = None
        self.p_w = None
        self.p = None
        self.clipped_p = None
        # self.build()

    def pointer_attention(self, encoder_features, decoder_features, encoder_mask, decoder_mask):
        """
        attention layer for pointer
        :param encoder_features: shape=(batch_size, seq_length, hidden_dim)
        :param decoder_features: shape=(batch_size, seq_length, hidden_dim)
        :param encoder_mask:
        :param decoder_mask:
        :return:
        """
        with tf.variable_scope('PointerAttention'):
            if self.attn_W_encoder is None:
                initializer = get_initializer(self.config.pointer_initializer, **self.config.__dict__)
                self.attn_W_encoder = tf.get_variable(name='attn_Wencoder',
                                                      shape=[self.config.hidden_size, self.config.hidden_size],
                                                      dtype=tf.float32,
                                                      initializer=initializer)
            if self.attn_W_decoder is None:
                initializer = get_initializer(self.config.pointer_initializer, **self.config.__dict__)
                self.attn_W_decoder = tf.get_variable(name='attn_Wdecoder',
                                                      shape=[self.config.hidden_size, self.config.hidden_size],
                                                      dtype=tf.float32,
                                                      initializer=initializer)
            if self.attn_V is None:
                initializer = get_initializer(self.config.pointer_initializer, **self.config.__dict__)
                self.attn_V = tf.get_variable(name='attn_V',
                                              shape=[self.config.hidden_size],
                                              dtype=tf.float32,
                                              initializer=initializer)
            # shape=(batch_size, dec_seq_length, hidden_dim)
            decoder_features = tf.tensordot(decoder_features, self.attn_W_decoder, axes=[2, 0])
            # shape=(batch_size, dec_seq_length, 1, hidden_dim)
            decoder_features = tf.expand_dims(decoder_features, axis=2)

            # shape=(batch_size, end_seq_length, hidden_dim)
            encoder_features = tf.tensordot(encoder_features, self.attn_W_encoder, axes=[2, 0])
            # shape=(batch_size, 1, enc_seq_length, hidden_dim)
            encoder_features = tf.expand_dims(encoder_features, axis=1)

            # shape=(batch_size, decoder_length, encoder_length, hidden_dim)
            tanh = tf.tanh(encoder_features + decoder_features)
            output = tf.tensordot(tanh, self.attn_V, axes=[3, 0])  # shape=(batch_size, decoder_length, encoder_length)
            with tf.variable_scope('mask'):
                # shape=(batch_size, decoder_length, encoder_length)
                mask = tf.expand_dims(decoder_mask, axis=2) * tf.expand_dims(encoder_mask, axis=1)
                adder = (1.0 - tf.cast(mask, tf.float32)) * -1e8

            # return output + adder
            return tf.nn.softmax(output + adder, axis=2)  # shape=(batch_size, decoder_length, encoder_length)

    def build(self, st, y_input=None, y_mask=None, encoder_outputs=None, encoder_mask=None, x_extend=None,
              oov_size=None, attention_prob=None, total_st=None, total_st_mask=None,
              use_pointer=False, reuse=tf.AUTO_REUSE):
        """
        predictions for each word in vocab and src(if use_pointer).
        :param st: shape=(batch_size, dec_seq_length, hidden_dim)
        :param y_input: shape=(batch_size, dec_seq_length, hidden_dim)
        :param y_mask: shape=(batch_size, seq_length)
        :param encoder_outputs: shape=(batch_size, seq_length, hidden_dim)
        :param encoder_mask: shape=(batch_size, seq_length)
        :param x_extend: shape=(batch_size, seq_length)
        :param oov_size: shape=(batch_size)
        :param attention_prob:
        :param use_pointer:
        :param reuse:
        :return: shape=(batch_size, seq_length, vocab_size/extended_vocab_size)
        """
        if use_pointer:
            assert x_extend is not None
            assert oov_size is not None

            # x_extend = x_extend[:, 1:]
            # encoder_outputs = encoder_outputs[:, 1:]
            # encoder_mask = encoder_mask[:, 1:]

        batch_size = get_shape_list(st)[0]
        with tf.variable_scope('Pointer', reuse=reuse):
            if use_pointer:
                self.W_c = tf.get_variable(name='W_c',
                                           shape=[self.config.hidden_size], dtype=tf.float32,
                                           initializer=get_initializer(self.config.pointer_initializer,
                                                                       **self.config.__dict__))
                self.W_s = tf.get_variable(name='W_s',
                                           shape=[self.config.hidden_size], dtype=tf.float32,
                                           initializer=get_initializer(self.config.pointer_initializer,
                                                                       **self.config.__dict__))
                self.W_y = tf.get_variable(name='W_y',
                                           shape=[self.config.hidden_size], dtype=tf.float32,
                                           initializer=get_initializer(self.config.pointer_initializer,
                                                                       **self.config.__dict__))
                if self.config.coverage:
                    self.W_h = tf.get_variable(name='W_h',
                                               shape=[self.config.hidden_size], dtype=tf.float32,
                                               initializer=get_initializer(self.config.pointer_initializer,
                                                                           **self.config.__dict__))
                self.b_ptr = tf.get_variable(name='b_ptr',
                                             shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

            # shape should be (batch_size, dec_seq_length, enc_seq_length)
            if use_pointer and attention_prob is None:
                self.encoder_attn = self.pointer_attention(encoder_outputs, st, encoder_mask, y_mask)
            else:
                self.encoder_attn = attention_prob

            inputs = [st]
            if self.config.coverage:
                inputs.append(tf.layers.dense(y_input, self.config.hidden_size,
                                              kernel_initializer=create_initializer(self.config.initializer_range),
                                              name='y_input'))

                assert total_st_mask is not None
                history_attention = attention_layer(
                    from_tensor=st,
                    to_tensor=y_input,
                    attention_mask=total_st_mask,
                    num_attention_heads=self.config.num_attention_heads,
                    size_per_head=self.config.hidden_size // self.config.num_attention_heads,
                    attention_probs_dropout_prob=self.parent_model.bert_attention_probs_dropout_prob,
                    initializer_range=self.config.initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_seq_length=self.config.decoder_seq_length,
                    to_seq_length=self.config.decoder_seq_length,
                    trainable=True,
                    masked=True)
                inputs.append(history_attention)
            if encoder_outputs is None:
                softmax_in = tf.concat(inputs, axis=-1)
                if use_pointer:
                    self.p_gen = tf.nn.sigmoid(tf.tensordot(st, self.W_s, axes=[2, 0]) +
                                               tf.tensordot(y_input, self.W_y, axes=[2, 0]) +
                                               self.b_ptr)
            else:
                if use_pointer:
                    # calc ct
                    # shape=(batch_size, dec_seq_length, enc_seq_length, 1)
                    weights = tf.expand_dims(self.encoder_attn, axis=-1)
                    enc = tf.expand_dims(encoder_outputs, axis=1)  # shape=(batch_size, 1, enc_seq_length, hidden_dim)
                    t = tf.multiply(weights, enc)  # shape=(batch_size, dec_seq_length, enc_seq_length, hidden_dim)
                    ct = tf.reduce_sum(t, axis=2)  # shape=(batch_size, dec_seq_length, hidden_dim)
                    inputs.append(ct)

                    p_gen_inputs = self.b_ptr \
                                   + tf.tensordot(ct, self.W_c, axes=[2, 0]) \
                                   + tf.tensordot(st, self.W_s, axes=[2, 0]) \
                                   + tf.tensordot(y_input, self.W_y, axes=[2, 0])
                    if self.config.coverage:
                        p_gen_inputs += tf.tensordot(history_attention, self.W_h, axes=[2, 0])
                    self.p_gen = tf.nn.sigmoid(p_gen_inputs)
                softmax_in = tf.concat(inputs, axis=-1)

            # shape=(batch_size, seq_length, vocab_size)
            # e_vocab = tf.layers.dense(st, self.config.vocab_size)
            # p_vocab = tf.nn.softmax(e_vocab, axis=-1)
            # p1 = tf.layers.dense(softmax_in, self.config.hidden_size)
            self.p_vocab = e_vocab = tf.nn.softmax(tf.layers.dense(softmax_in, self.config.vocab_size), axis=-1)

            if use_pointer:
                # extend p_vocab with extra zeros.
                max_oov_size = tf.reduce_max(oov_size)
                extra_zeros = tf.zeros(shape=(batch_size, self.config.decoder_seq_length, max_oov_size))
                p_vocab_extend = tf.concat((self.p_vocab, extra_zeros), axis=2)

                # create encoder_attn matrix.
                with tf.variable_scope('projection'):
                    i = tf.tile(tf.expand_dims(x_extend, axis=1), [1, self.config.decoder_seq_length, 1])
                    i1, i2 = tf.meshgrid(tf.range(batch_size),
                                         tf.range(self.config.decoder_seq_length), indexing="ij")
                    i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, self.config.encoder_seq_length])
                    i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, self.config.encoder_seq_length])
                    # Create final indices
                    idx = tf.stack([i1, i2, i], axis=-1)
                    # Output shape
                    to_shape = [batch_size, self.config.decoder_seq_length, self.config.vocab_size + max_oov_size]
                    # Get scattered tensor
                    self.p_w = tf.scatter_nd(idx, self.encoder_attn, to_shape)

                p_gen = tf.expand_dims(self.p_gen, axis=-1)
                self.p = p_gen * p_vocab_extend + (1 - p_gen) * self.p_w

                self.clipped_p = self.p + EPSILON
                return self.clipped_p
            else:
                return self.p_vocab + EPSILON


class Model:
    def __init__(self, config, data=None, copy_config=True, multi_gpu_mode=False):
        if copy_config:
            self.config = copy.deepcopy(config)
        else:
            self.config = config
        self.data = data
        self.multi_gpu_mode = multi_gpu_mode

        self.dict = dict()

        self.encoder = None
        self.decoder = None
        self.sequence_output = None
        self.pointer = None
        self.p = None
        self.loss_ml = None
        self.loss_rl = None
        self.loss_matrix_ml = None
        self.loss_matrix_rl = None
        self.loss = None
        self.y_pred = None
        self.bert_grad_and_vars = None
        self.other_grad_and_vars = None
        self.optimization = None
        self.update_step = None

    def make_placeholders(self):
        decoder_seq_length = self.config.decoder_seq_length
        encoder_seq_length = self.config.encoder_seq_length
        encoder_output_length = self.config.encoder_output_length

        if not self.multi_gpu_mode:
            self.global_step = tf.get_variable(name='global_step', shape=[], dtype=tf.int64, trainable=False,
                                               initializer=tf.zeros_initializer())
        self.bert_lr = tf.placeholder(dtype=tf.float32, shape=[], name='bert_lr')
        self.other_lr = tf.placeholder(dtype=tf.float32, shape=[], name='other_lr')

        self.y_ids = tf.placeholder(dtype=tf.int32, shape=[None, decoder_seq_length], name='y_ids')
        self.y_ids_loss = tf.placeholder(dtype=tf.int32, shape=[None, decoder_seq_length], name='y_ids_loss')
        self.y_extend = tf.placeholder(dtype=tf.int32, shape=[None, decoder_seq_length], name='y_extend')
        self.y_mask = tf.placeholder(dtype=tf.int32, shape=[None, decoder_seq_length], name='y_mask')
        self.x_ids = tf.placeholder(dtype=tf.int32, shape=[None, encoder_seq_length], name='x_ids')
        self.x_extend = tf.placeholder(dtype=tf.int32, shape=[None, encoder_seq_length], name='x_extend')
        self.x_mask = tf.placeholder(dtype=tf.int32, shape=[None, encoder_seq_length], name='x_mask')
        self.oov_size = tf.placeholder(dtype=tf.int32, shape=[None], name='oov_size')
        self.encoder_output_input_layers = [tf.placeholder(dtype=tf.float32,
                                                           shape=[None, encoder_output_length, self.config.hidden_size],
                                                           name='encoder_output_input_%d' % i)
                                            for i in range(self.config.num_hidden_layers)]  # TODO
        self.encoder_output_input = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, encoder_output_length, self.config.hidden_size],
                                                   name='encoder_output_input')  # TODO

        self.bert_hidden_dropout_prob = tf.placeholder(dtype=tf.float32, shape=[], name='hidden_dropout_prob')
        self.bert_attention_probs_dropout_prob = tf.placeholder(dtype=tf.float32, shape=[],
                                                                name='attention_probs_dropout_prob')

    def forward(self, is_training=True):
        print('Building Encoder...')

        # token_type_ids: [1, ..., 1, 2, ..., 2, ..., 15, ..., 15]
        token_type_ids = []
        for i in range(self.config.type_vocab_size):
            token_type_ids += ([i + 1] * (self.config.segment_length + 1))
        token_type_ids = np.array(token_type_ids)
        token_type_ids = tf.constant(token_type_ids)

        self.encoder = BertModel(
            config=self.config,
            # is_training=is_training,
            input_ids=self.x_ids,
            input_mask=self.x_mask,
            token_type_ids=token_type_ids,  # !!!!! : add token_type_ids
            use_one_hot_embeddings=False,
            hidden_dropout_prob=self.bert_hidden_dropout_prob,
            attention_probs_dropout_prob=self.bert_attention_probs_dropout_prob,
            scope='enc',
            encoder_output=None,
            encoder_mask=None,
            trainable_layers=self.config.encoder_trainable_layers,
            embedding_trainable=self.config.embedding_trainable,
            pooler_layer_trainable=False,
            masked_layer_trainable=False,
            attention_layer_trainable=False
        )
        # shape = [batch_size, enc_seq_len, hidden_size]
        self.encoder_output = self.encoder.get_sequence_output()
        self.encoder_mask = self.encoder.get_sequence_mask() # TODO : add sequence_mask
        # self.distance = self.encoder.get_distance()
        self.topk = self.encoder.get_topk()
        # self.encoder_output_before = self.encoder.get_encoder_output_before()
        # self.encoder_output_after = self.encoder.get_encoder_output_after()
        self.embedding_output = self.encoder.get_embedding()

        self.encoder_output_for_decoder = self.encoder.all_encoder_layers if self.config.align_layers else self.encoder.get_sequence_output()

        print('Building Decoder...')
        if self.config.align_layers:
            encoder_output_param = self.encoder.all_encoder_layers if is_training else self.encoder_output_input_layers
        else:
            encoder_output_param = self.encoder_output if is_training else self.encoder_output_input

        self.decoder = BertModel(
            config=self.config,
            # is_training=is_training,
            input_ids=self.y_ids,
            input_mask=self.y_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False,
            hidden_dropout_prob=self.bert_hidden_dropout_prob,
            attention_probs_dropout_prob=self.bert_attention_probs_dropout_prob,
            scope='dec',
            encoder_output=encoder_output_param,
            encoder_mask=self.encoder_mask, # TODO:self.x_mask,
            trainable_layers=self.config.trainable_layers,
            embedding_trainable=self.config.embedding_trainable,
            pooler_layer_trainable=self.config.pooler_layer_trainable,
            masked_layer_trainable=self.config.masked_layer_trainable,
            attention_layer_trainable=self.config.attention_layer_trainable
        )
        self.decoder_output = self.decoder.sequence_output
        self.attention = self.decoder.all_attention_probs
        # TODO: get attention parameters

        print('Building Pointer...')
        # with tf.variable_scope('shift'):
        #     shifted_inputs = tf.concat([tf.zeros(dtype=tf.int32, shape=[1, 1]), self.y_ids], axis=1)[:, :-1]
        self.sequence_output = self.decoder.sequence_output  # shape=(batch_size, seq_length, hidden_dim)
        # self.attention_prob = self.decoder.attention_prob
        self.pointer = Pointer(self.config, self)
        '''
        build(self, st, y_extend=None, y_mask=None, encoder_outputs=None, encoder_mask=None, x_extend=None,
              oov_size=None, use_pointer=False):
        '''
        # p: shape=(batch_size, seq_length, vocab_size/extended)
        self.proba = self.p = self.pointer.build(st=self.sequence_output,
                                                 y_input=self.decoder.embedding_output,
                                                 y_mask=self.y_mask,
                                                 encoder_outputs=self.encoder_output if is_training else self.encoder_output_input,
                                                 encoder_mask=self.x_mask,
                                                 x_extend=self.x_extend,
                                                 oov_size=self.oov_size,
                                                 # attention_prob=self.attention_prob,
                                                 total_st=self.decoder.embedding_output,
                                                 total_st_mask=self.decoder.self_attention_mask,
                                                 use_pointer=self.config.use_pointer)
        # p = self.pointer.build(st=self.sequence_output,
        #                        use_pointer=False)
        self.y_pred = tf.argmax(self.p, axis=-1, name='y_pred')

    def calc_loss(self):
        print('Building Loss...')
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            vsize = get_shape_list(self.p, expected_rank=[3])[2]

            def loss_function(logits=None, labels=None):
                assert logits is not None, labels is not None
                mask = tf.expand_dims(tf.cast(self.y_mask, dtype=tf.float32), axis=-1)
                loss_ = tf.reduce_mean(-tf.reduce_sum(
                    tf.one_hot(self.y_ids_loss, depth=vsize) * tf.log(logits) * mask,
                    reduction_indices=[-1]))
                return loss_

            # base version
            # loss = loss_function(logits=self.p, labels=self.y_ids_loss)
            # keras version
            mask = tf.cast(self.y_mask, tf.float32)
            self.loss_matrix_ml = tf.keras.losses.categorical_crossentropy(
                y_true=tf.one_hot(self.y_ids_loss, depth=vsize),
                y_pred=self.p) * mask
            self.unstack_loss = tf.reduce_sum(self.loss_matrix_ml, axis=-1) / tf.reduce_sum(mask, axis=-1)
            # print('self.unstack_loss.shape:', self.unstack_loss.shape)
            loss = tf.reduce_mean(self.unstack_loss)
            # loss = tf.reduce_mean(self.loss_matrix_ml)

            # seq2seq version
            # loss = tf.contrib.seq2seq.sequence_loss(logits=self.p,
            #                                         targets=self.y_ids_loss,
            #                                         weights=tf.cast(self.y_mask, dtype=tf.float32),
            #                                         softmax_loss_function=loss_function)
            self.loss_ml = loss
            tf.summary.scalar(name='loss_ml', tensor=self.loss_ml)

        self.loss = self.loss_ml
        # self.loss = self.loss_rl

    def compute_gradients(self):
        print('Building Gradients Computation...')
        with tf.variable_scope('compute_gradients', reuse=tf.AUTO_REUSE):
            bert_vars = []
            other_vars = []
            for v in tf.trainable_variables():
                if v.name.startswith('enc') or v.name.startswith('dec') or v.name.startswith('embeddings'):
                    bert_vars.append(v)
                else:
                    other_vars.append(v)
            if len(bert_vars) > 0:
                self.dict['grads'] = grads = tf.gradients(self.loss, bert_vars + other_vars,
                                                          colocate_gradients_with_ops=False)
                for v, g in zip(bert_vars + other_vars, grads):
                    if g is None:
                        print('[Warning] [Gradients] None: %s' % v.name)
                    # else:
                    #     grads[i] = tf.Print(g, [g], message='[grads %d]' % i, summarize=99999999)
                clipped_bert_grads, _ = tf.clip_by_global_norm(grads[:len(bert_vars)], 1.0)
                self.bert_grad_and_vars = zip(clipped_bert_grads, bert_vars)

                # clipped_other_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in
                #                      zip(grads[len(bert_vars):], other_vars) if grad is not None]
                clipped_other_grads, _ = tf.clip_by_global_norm(grads[len(bert_vars):], 5.0)
                self.other_grad_and_vars = zip(clipped_other_grads, other_vars)

            else:
                other_optimizer = tf.train.AdamOptimizer(learning_rate=self.other_lr)
                self.dict['grads'] = grads = tf.gradients(self.loss, other_vars,
                                                          colocate_gradients_with_ops=False)
                # clipped_other_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in zip(grads, other_vars)
                #                      if grad is not None]
                clipped_other_grads = tf.clip_by_global_norm(grads, 5)
                self.other_grad_and_vars = zip(clipped_other_grads, other_vars)
                self.optimization = other_optimizer.apply_gradients(zip(clipped_other_grads, other_vars),
                                                                    global_step=self.global_step)

    def backward(self, grad_and_vars, optimizer=None):
        with tf.variable_scope('backward', reuse=tf.AUTO_REUSE):
            if optimizer is None:
                self.optimizer = optimizer = tf.train.AdamOptimizer(0.0001)
            train_op = optimizer.apply_gradients(grad_and_vars)
        return train_op

    def build(self, is_training=True):
        self.make_placeholders()
        self.forward()
        if is_training:
            self.calc_loss()
            self.compute_gradients()
            total_train_steps = self.config.epochs * self.config.steps_per_epoch
            train_ops = []
            if self.bert_grad_and_vars is not None:
                with tf.variable_scope('backward'):
                    self.bert_optimizer = create_bert_optimizer(config=self.config,init_lr=self.config.bert_learning_rate,
                                                                num_train_steps=total_train_steps,
                                                                warmup_proportion=0.04)
                bert_train_op = self.backward(grad_and_vars=self.bert_grad_and_vars,
                                              optimizer=self.bert_optimizer)
                train_ops.append(bert_train_op)
            if self.other_grad_and_vars is not None:
                with tf.variable_scope('backward'):
                    self.other_optimizer = create_other_optimizer(config=self.config,init_lr=self.config.other_learning_rate,
                                                                  num_train_steps=total_train_steps,
                                                                  warmup_proportion=0.04)
                other_train_op = self.backward(self.other_grad_and_vars,
                                               optimizer=self.other_optimizer)
                train_ops.append(other_train_op)

            train_ops.append(tf.assign_add(self.global_step, 1, name='update_step'))
            self.train_op = tf.group(*train_ops)

    def get_feed_dict(self, is_training, batch_data):
        y_token, y_ids, y_ids_loss, y_extend, y_mask, x_token, x_ids, x_extend, x_mask, oov_size, oovs = batch_data
        assert y_token.shape[0] == y_ids.shape[0] == y_ids_loss.shape[0] == y_extend.shape[0] == y_mask.shape[0] == \
               x_token.shape[0] == x_ids.shape[0] == x_extend.shape[0] == x_mask.shape[0] == \
               oov_size.shape[0] == oovs.shape[0]

        fd = dict()
        # fd[self.y_token] =   y_token
        fd[self.y_ids] = y_ids
        fd[self.y_ids_loss] = y_ids_loss
        # fd[self.y_extend] =  y_extend
        fd[self.y_mask] = y_mask
        # fd[self.x_token] =   x_token
        fd[self.x_ids] = x_ids
        fd[self.x_extend] = x_extend
        fd[self.x_mask] = x_mask
        fd[self.oov_size] = oov_size
        # fd[self.oovs] =      oovs
        fd[self.bert_hidden_dropout_prob] = self.config.hidden_dropout_prob if is_training else 0.0
        fd[self.bert_attention_probs_dropout_prob] = self.config.attention_probs_dropout_prob if is_training else 0.0

        return fd


class MultiGPUModel:
    def __init__(self, config, num_gpus, copy_config=True):
        if copy_config:
            self.config = copy.deepcopy(config)
        else:
            self.config = config
        self.num_gpus = num_gpus
        self.models = []
        self.bert_grad_and_vars = []
        self.other_grad_and_vars = []
        self.averaged_bert_grad_and_vars = None
        self.averaged_other_grad_and_vars = None
        self.global_step = tf.get_variable(name='global_step', shape=[], dtype=tf.int64, trainable=False,
                                           initializer=tf.zeros_initializer())
        for i in range(num_gpus):
            self.models.append(Model(self.config, copy_config=False, multi_gpu_mode=True))

    @classmethod
    def average_gradients(cls, tower_grads):
        avg_grads = []

        # list all the gradient obtained from different GPU
        # grad_and_vars represents gradient of w1, b1, w2, b2 of different gpu respectively
        for grad_and_vars in zip(*tower_grads):  # w1, b1, w2, b2
            # calculate average gradients
            # print('grad_and_vars: ', grad_and_vars)
            grads = []
            for g, _ in grad_and_vars:  # different gpu
                expanded_g = tf.expand_dims(g, 0)  # expand one dimension (5, 10) to (1, 5, 10)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)  # for 4 gpu, 4 (1, 5, 10) will be (4, 5, 10),concat the first dimension
            grad = tf.reduce_mean(grad, 0)  # calculate average by the first dimension
            # print('grad: ', grad)

            v = grad_and_vars[0][1]  # get w1 and then b1, and then w2, then b2, why?
            # print('v',v)
            grad_and_var = (grad, v)
            # print('grad_and_var: ', grad_and_var)
            # corresponding variables and gradients
            avg_grads.append(grad_and_var)
        return avg_grads

    def build(self, is_training=True):
        print('Building Multi-GPU Model with %d GPUs' % self.num_gpus)
        losses = []
        for i, model in enumerate(self.models):
            with tf.device('/gpu:%d' % i):
                print('Building Model on GPU %d' % i)
                model.make_placeholders()
                model.forward(is_training)
                model.calc_loss()
                if is_training:
                    model.compute_gradients()
            losses.append(model.unstack_loss)
            if is_training:
                self.bert_grad_and_vars.append(model.bert_grad_and_vars)
                self.other_grad_and_vars.append(model.other_grad_and_vars)
        # with tf.device('/cpu'):
        self.encoder_output = tf.concat([model.encoder_output for model in self.models], axis=0)
        if self.config.align_layers:
            self.encoder_output_for_decoder = []
            for i in range(self.config.num_hidden_layers):
                self.encoder_output_for_decoder.append(
                    tf.concat([model.encoder.all_encoder_layers[i] for model in self.models], axis=0))
        else:
            self.encoder_output_for_decoder = self.encoder_output
        self.y_pred = tf.concat([model.y_pred for model in self.models], axis=0)
        self.loss_matrix_ml = tf.concat([model.loss_matrix_ml for model in self.models], axis=0)
        stacked_loss = tf.concat(losses, axis=0)
        # print('Multi-GPU Stacked Loss Shape:', stacked_loss.shape)
        self.loss = tf.reduce_mean(stacked_loss, axis=0)
        # print('Multi-GPU Loss Shape:', self.loss.shape)
        if is_training:
            self.averaged_bert_grad_and_vars = self.average_gradients(self.bert_grad_and_vars)
            self.averaged_other_grad_and_vars = self.average_gradients(self.other_grad_and_vars)

            total_train_steps = self.config.epochs * self.config.steps_per_epoch
            train_ops = []
            if self.bert_grad_and_vars is not None:
                self.bert_optimizer = create_bert_optimizer(config=self.config,init_lr=self.config.bert_learning_rate,
                                                            num_train_steps=total_train_steps,
                                                            warmup_proportion=0.1)
            if self.other_grad_and_vars is not None:
                self.other_optimizer = create_other_optimizer(config=self.config,init_lr=self.config.other_learning_rate,
                                                              num_train_steps=total_train_steps,
                                                              warmup_proportion=0.1)

            model = self.models[0]
            with tf.device('/gpu:0'):
                if self.bert_grad_and_vars is not None:
                    bert_train_op = model.backward(grad_and_vars=self.averaged_bert_grad_and_vars,
                                                   optimizer=self.bert_optimizer)
                    train_ops.append(bert_train_op)
                if self.other_grad_and_vars is not None:
                    other_train_op = model.backward(grad_and_vars=self.averaged_other_grad_and_vars,
                                                    optimizer=self.other_optimizer)
                    train_ops.append(other_train_op)

            train_ops.append(tf.assign_add(self.global_step, 1, name='update_step'))
            self.train_op = tf.group(*train_ops)

    def get_decode_encoder_feed_dict(self, batch_data, is_predict=False):
        assert batch_data[0].shape[0] >= self.num_gpus
        for i, data in enumerate(batch_data):
            if data is not None and type(data) != list:
                batch_data[i] = np.array_split(data, self.num_gpus, axis=0)
        if is_predict:
            x_token, x_ids, x_extend, x_mask, oov_size, oovs = batch_data
        else:
            y_token, y_ids, y_ids_loss, y_extend, y_mask, x_token, x_ids, x_extend, x_mask, oov_size, oovs = batch_data
        # length = y_token.shape[0]
        # assert y_token.shape[0] == y_ids.shape[0] == y_ids_loss.shape[0] == y_extend.shape[0] == y_mask.shape[0] == \
        #                 x_token.shape[0] == x_ids.shape[0] == x_extend.shape[0] == x_mask.shape[0] == \
        #                 oov_size.shape[0] == oovs.shape[0]
        # assert length >= self.num_gpus
        fd = dict()
        for i, model in enumerate(self.models):
            fd[model.x_ids] = x_ids[i]
            fd[model.x_mask] = x_mask[i]
            fd[model.bert_hidden_dropout_prob] = 0.0
            fd[model.bert_attention_probs_dropout_prob] = 0.0
        return fd

    def _split_encoder_output(self, encoder_output):
        if self.config.align_layers:
            split = []
            for i in range(self.config.num_hidden_layers):
                split.append(np.array_split(encoder_output[i], self.num_gpus, axis=0))
        else:
            split = np.array_split(encoder_output, self.num_gpus, axis=0)
        return split

    def get_decode_decoder_feed_dict(self, batch_data, split_encoder_output, is_predict=False, decoder_seq_length=None):
        assert type(batch_data[0]) == list or batch_data[0].shape[0] >= self.num_gpus
        for i, data in enumerate(batch_data):
            if data is not None and type(data) != list:
                batch_data[i] = np.array_split(data, self.num_gpus, axis=0)
        if is_predict:
            y_ids, x_token, x_ids, x_extend, x_mask, oov_size, oovs = batch_data
        else:
            y_token, y_ids, y_ids_loss, y_extend, y_mask, x_token, x_ids, x_extend, x_mask, oov_size, oovs = batch_data
        # length = y_token.shape[0]
        # assert y_token.shape[0] == y_ids.shape[0] == y_ids_loss.shape[0] == y_extend.shape[0] == y_mask.shape[0] == \
        #                 x_token.shape[0] == x_ids.shape[0] == x_extend.shape[0] == x_mask.shape[0] == \
        #                 oov_size.shape[0] == oovs.shape[0]
        # assert length >= self.num_gpus
        fd = dict()
        for i, model in enumerate(self.models):
            if self.config.align_layers:
                for layer_index in range(self.config.num_hidden_layers):
                    fd[model.encoder_output_input_layers[layer_index]] = split_encoder_output[layer_index][i]
            else:
                fd[model.encoder_output_input] = split_encoder_output[i]
            # fd[model.y_token] =   y_token[i]
            fd[model.y_ids] = y_ids[i]
            if not is_predict:
                fd[model.y_ids_loss] = y_ids_loss[i]
                # fd[model.y_extend] =  y_extend[i]
                fd[model.y_mask] = y_mask[i]
            else:
                k = np.ones(shape=(y_ids[0].shape[0], decoder_seq_length))
                # print(k.shape)
                fd[model.y_mask] = k
            # fd[model.x_token] =   x_token[i]
            fd[model.x_ids] =       x_ids[i]
            fd[model.x_extend] = x_extend[i]
            fd[model.x_mask] = x_mask[i]
            fd[model.oov_size] = oov_size[i]
            # fd[model.oovs] =      oovs[i]
            fd[model.bert_hidden_dropout_prob] = 0.0
            fd[model.bert_attention_probs_dropout_prob] = 0.0

        return fd

    def get_feed_dict(self, is_training, batch_data):
        assert batch_data[0].shape[0] >= self.num_gpus
        for i, data in enumerate(batch_data):
            if data is not None and type(data) != list:
                batch_data[i] = np.array_split(data, self.num_gpus, axis=0)
        y_token, y_ids, y_ids_loss, y_extend, y_mask, x_token, x_ids, x_extend, x_mask, oov_size, oovs = batch_data
        # length = y_token.shape[0]
        # assert y_token.shape[0] == y_ids.shape[0] == y_ids_loss.shape[0] == y_extend.shape[0] == y_mask.shape[0] == \
        #                 x_token.shape[0] == x_ids.shape[0] == x_extend.shape[0] == x_mask.shape[0] == \
        #                 oov_size.shape[0] == oovs.shape[0]
        # assert length >= self.num_gpus
        fd = dict()
        for i, model in enumerate(self.models):
            # fd[model.y_token] =   y_token[i]
            fd[model.y_ids] = y_ids[i]
            fd[model.y_ids_loss] = y_ids_loss[i]
            # fd[model.y_extend] =  y_extend[i]
            fd[model.y_mask] = y_mask[i]
            # fd[model.x_token] =   x_token[i]
            fd[model.x_ids] = x_ids[i]
            fd[model.x_extend] = x_extend[i]
            fd[model.x_mask] = x_mask[i]
            fd[model.oov_size] = oov_size[i]
            # fd[model.oovs] =      oovs[i]
            fd[model.bert_hidden_dropout_prob] = self.config.hidden_dropout_prob if is_training else 0.0
            fd[
                model.bert_attention_probs_dropout_prob] = self.config.attention_probs_dropout_prob if is_training else 0.0
        return fd
