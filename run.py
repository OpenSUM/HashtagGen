import gc
import os
import time
import traceback

import numpy as np
import tensorflow as tf

from constants import *
from model import MultiGPUModel
from utils import calc_rouge, eprint, BertConfig
from utils.Batcher import get_batcher, get_predict_batcher
from utils.Saver import Saver
from utils.data_loader import load_vocab, id2text
from constants import SEP_TOKEN, UNK_TOKEN


def test_batch(model, sess, eval_batcher, seq_length, word2id, id2word, use_pointer, substr_prefix, verbose=False, **kwargs):
    assert len(word2id) == len(id2word)
    vocab_size = len(word2id)
    outputs = []
    trues = []
    sources = []
    losses = []
    substr_replacement = ' {}'.format(substr_prefix)
    # for batch_index, (y_token, y_ids, y_ids_loss, y_extend, y_mask,
    #                   x_token, x_ids, x_extend, x_mask, oov_size, oovs) in enumerate(eval_batcher.batch(), 1):
    start_time = time.time()
    last_time = start_time
    for batch_index, batch_data in enumerate(eval_batcher.batch(), 1):
        (y_token, y_ids, y_ids_loss, y_extend, y_mask,
         x_token, x_ids, x_extend, x_mask, oov_size, oovs) = batch_data
        if not use_pointer:
            y_ids_loss[y_ids_loss >= vocab_size] = word2id[UNK_TOKEN]

        # run encoder
        fd = model.get_decode_encoder_feed_dict(batch_data)
        # TODO
        encoder_output = sess.run(model.encoder_output_for_decoder, feed_dict=fd)
        encoder_output = model._split_encoder_output(encoder_output)

        # run decoder step by step
        prev_extend = np.zeros(shape=y_ids.shape, dtype=np.int32)
        prev_ids = np.zeros(shape=y_ids.shape, dtype=np.int32)
        # not_finish = set(range(y_ids.shape[0]))
        batch_losses = []
        for i in range(seq_length):
            batch_data[1] = prev_ids
            fd = model.get_decode_decoder_feed_dict(batch_data=batch_data, split_encoder_output=encoder_output)
            result = sess.run([model.y_pred, model.loss_matrix_ml], feed_dict=fd)
            preds, loss = result[0], result[1]
            batch_losses.append(loss[:, i])
            prev_extend = np.concatenate((preds[:, :i + 1], prev_extend[:, i + 1:]), axis=-1)
            prev_ids = np.copy(prev_extend)
            prev_ids[prev_ids >= vocab_size] = word2id[UNK_TOKEN]

            # 记录已经生成SEP标签的句子，如果所有句子都已经生成SEP标签，则提前结束。
            # delete = set()
            # for index in not_finish:
            #     if word2id[SEP_TOKEN] in preds[index, :i + 1]:
            #         delete.add(index)
            # not_finish -= delete
            # if len(not_finish) == 0:
            #     break

        batch_loss = np.vstack(batch_losses).T
        batch_loss = np.sum(batch_loss, axis=-1) / np.sum(y_mask, axis=-1, dtype=np.float32)
        losses.append(batch_loss)
        for i, abs in enumerate(prev_extend.tolist()):
            output = []
            for w in abs:
                if w != word2id[SEP_TOKEN]:
                    if w > 0:
                        output.append(w)
                else:
                    break
            if len(output) == 0:
                output = [word2id[SEP_TOKEN]]
            outputs.append(id2text(ids=output, id2word=id2word, oov=oovs[i]).replace(substr_replacement, ''))
        tmp_trues = [' '.join(l).replace(substr_replacement, '') for l in y_token.tolist()]
        if verbose:
            for t in tmp_trues:
                print(t)
        trues.extend(tmp_trues)
        sources.extend([' '.join(l).replace(substr_replacement, '') for l in x_token.tolist()])
        t = time.time()
        print('Batch {}, time: {:.2f}s, total time: {:.2f}s'.format(batch_index, t - last_time, t - start_time))
        last_time = t

    print('Total Eval Time: {:.2f}s'.format(time.time() - start_time))
    scores = calc_rouge(outputs, trues)
    return scores, np.mean(np.concatenate(losses)), dict(source=sources, ref=trues, cand=outputs)


def test_batch_when_training(model, sess, eval_batcher, seq_length, word2id, id2word, use_pointer, substr_prefix, **kwargs):
    assert len(word2id) == len(id2word)
    vocab_size = len(word2id)
    outputs = []
    trues = []
    sources = []
    losses = []
    substr_replacement = ' {}'.format(substr_prefix)
    # for batch_index, (y_token, y_ids, y_ids_loss, y_extend, y_mask,
    #                   x_token, x_ids, x_extend, x_mask, oov_size, oovs) in enumerate(eval_batcher.batch(), 1):
    start_time = time.time()
    for batch_index, batch_data in enumerate(eval_batcher.batch(), 1):
        (y_token, y_ids, y_ids_loss, y_extend, y_mask,
         x_token, x_ids, x_extend, x_mask, oov_size, oovs) = batch_data
        if not use_pointer:
            y_ids_loss[y_ids_loss >= vocab_size] = word2id[UNK_TOKEN]
        prev_extend = np.zeros(shape=y_ids.shape, dtype=np.int32)
        prev_ids = np.zeros(shape=y_ids.shape, dtype=np.int32)
        # not_finish = set(range(y_ids.shape[0]))
        batch_losses = []
        flag = True
        for i in range(seq_length):
            fd = model.get_feed_dict(is_training=False, batch_data=batch_data)
            if flag:
                start_time = time.time()
                flag = False
            result = sess.run([model.y_pred, model.loss_matrix_ml], feed_dict=fd)
            preds, loss = result[0], result[1]
            batch_losses.append(loss[:, i])
            prev_extend = np.concatenate((preds[:, :i + 1], prev_extend[:, i + 1:]), axis=-1)
            prev_ids = np.copy(prev_extend)
            prev_ids[prev_ids >= vocab_size] = word2id[UNK_TOKEN]

            # 记录已经生成SEP标签的句子，如果所有句子都已经生成SEP标签，则提前结束。
            # delete = set()
            # for index in not_finish:
            #     if word2id[SEP_TOKEN] in preds[index, :i + 1]:
            #         delete.add(index)
            # not_finish -= delete
            # if len(not_finish) == 0:
            #     break

        batch_loss = np.vstack(batch_losses).T
        # exist = batch_loss != 0.0
        # den = exist.sum(axis=-1)
        batch_loss = np.sum(batch_loss, axis=-1) / np.sum(y_mask, axis=-1, dtype=np.float32)
        losses.append(batch_loss)
        for i, abs in enumerate(prev_extend.tolist()):
            output = []
            for w in abs:
                if w != word2id[SEP_TOKEN]:
                    if w > 0:
                        output.append(w)
                else:
                    break
            if len(output) == 0:
                output = [word2id[SEP_TOKEN]]
            outputs.append(id2text(ids=output, id2word=id2word, oov=oovs[i]).replace(substr_replacement, ''))
        trues.extend([' '.join(l).replace(substr_replacement, '') for l in y_token.tolist()])
        sources.extend([' '.join(l).replace(substr_replacement, '') for l in x_token.tolist()])

    print('Eval Time: {:.2f}s'.format(time.time() - start_time))
    scores = calc_rouge(outputs, trues)
    return scores, np.mean(np.concatenate(losses)), dict(source=sources, ref=trues, cand=outputs)


def test_batch_with_beam_search(model, sess, eval_batcher, seq_length, word2id, id2word, use_pointer, beam_size=2, **kwargs):
    assert len(word2id) == len(id2word)
    vocab_size = len(word2id)
    # todo: this function needs to be modified.
    outputs = []
    trues = []
    sources = []
    for batch_index, (y_token, y_ids, y_ids_loss, y_extend, y_mask,
                      x_token, x_ids, x_extend, x_mask, oov_size, oovs) in enumerate(eval_batcher.batch(batch_size=1),
                                                                                     1):
        if not use_pointer:
            y_ids_loss[y_ids_loss >= vocab_size] = word2id[UNK_TOKEN]
        candidate_inputs = [np.zeros(shape=y_ids.shape[1:], dtype=np.int32)]
        candidate_extends = [np.zeros(shape=y_ids.shape[1:], dtype=np.int32)]
        candidate_probas = [0.0]
        # not_finish = set(range(y_ids.shape[0]))
        batch_losses = []
        finish = []
        for i in range(seq_length):
            tmp_inputs = []
            tmp_extends = []
            tmp_probas = []
            for candidate_input, candidate_extend, candidate_proba in zip(candidate_inputs, candidate_extends,
                                                                          candidate_probas):
                fd = {
                    model.y_ids: candidate_input.reshape(1, *candidate_input.shape),
                    model.y_ids_loss: y_ids_loss,
                    # model.y_extend: prev_extend,
                    model.y_mask: y_mask,
                    model.x_ids: x_ids,
                    model.x_extend: x_extend,
                    # model.x_output: x_output,
                    model.x_mask: x_mask,
                    model.oov_size: oov_size,
                    # model.lr: learning_rate,
                    model.bert_hidden_dropout_prob: 0,
                    model.bert_attention_probs_dropout_prob: 0,
                }
                result = sess.run([model.p, model.loss_matrix_ml], feed_dict=fd)
                proba, loss = result[0], result[1]  # shape of proba = (batch_size, dec_seq_length, extend_vocab_size)
                # proba = proba[:, i, :]       # shape of proba = (batch_size, extend_vocab_size)
                proba = np.log(proba.reshape(proba.shape[1:]))
                arg = np.argsort(-proba, axis=-1)  # (batch, dec_seq_length, extend_vocab_size)
                for j in range(beam_size):
                    # batch_losses.append(loss[:, i])
                    this_extend = np.concatenate((arg[:i + 1, j], candidate_extend[i + 1:]), axis=-1)
                    this_proba = candidate_proba + proba[i, arg[i, j]]

                    if arg[i, j] == word2id[SEP_TOKEN]:
                        finish.append((this_proba, this_extend))
                    else:
                        this_input = np.copy(this_extend)
                        this_input[this_input >= vocab_size] = word2id[UNK_TOKEN]

                        tmp_inputs.append(this_input)
                        tmp_extends.append(this_extend)
                        tmp_probas.append(this_proba)

                    # prev_extend = np.concatenate((preds[:, :i + 1], prev_extend[:, i + 1:]), axis=-1)
                    # prev_ids = np.copy(prev_extend)
                    # prev_ids[prev_ids >= VOCAB_SIZE] = UNK_ID
            # tmp_inputs = [np.vsplit(a, a.shape[0]) for a in tmp_inputs]
            # tmp_extends = [np.vsplit(a, a.shape[0]) for a in tmp_extends]
            # tmp_probas = [np.vsplit(a, a.shape[0]) for a in tmp_probas]
            # pqs = [PQ(beam_size) for _ in range(beam_size)]
            # for i, pq in enumerate(pqs):
            # pq = PQ(beam_size)
            # for inp, ext, proba in zip(tmp_inputs, tmp_extends, tmp_probas):
            #     if pq.full():
            #         pq.get()
            #     pq.put((proba, _count, inp, ext))
            #     _count += 1

            candidate_inputs, candidate_extends, candidate_probas = [], [], []
            iterator = sorted(zip(tmp_inputs, tmp_extends, tmp_probas), key=lambda x: x[2], reverse=True)[:beam_size]
            for inp, ext, proba in iterator:
                # probas, inps, exts = [], [], []
                # for pq in pqs:
                # proba, _, inp, ext = pq.get()
                # probas.append(proba)
                # inps.append(inp)
                # exts.append(ext)
                candidate_inputs.append(inp)
                candidate_extends.append(ext)
                candidate_probas.append(proba)
            # candidate_inputs, candidate_extends, candidate_probas = tmp_inputs, tmp_extends, tmp_probas

            # 记录已经生成SEP标签的句子，如果所有句子都已经生成SEP标签，则提前结束。
            # delete = set()
            # for index in not_finish:
            #     if word2id[SEP_TOKEN] in preds[index, :i + 1]:
            #         delete.add(index)
            # not_finish -= delete
            # if len(not_finish) == 0:
            #     break

        iterator = sorted(zip(candidate_inputs, candidate_extends, candidate_probas), key=lambda x: x[2], reverse=True)[
                   :beam_size]
        for inp, ext, proba in iterator:
            finish.append((proba, ext))
        finish.sort(key=lambda x: x[0], reverse=True)
        output = []
        for w in finish[0][1].tolist():
            if w != word2id[SEP_TOKEN]:
                if w > 0:
                    output.append(w)
            else:
                break
        if len(output) == 0:
            output = [word2id[SEP_TOKEN]]
        outputs.append(id2text(ids=output, id2word=id2word, oov=oovs.reshape(oovs.shape[1:])))
        trues.append(' '.join(y_token[0]))
        sources.append(' '.join(x_token[0]))

    scores = calc_rouge(outputs, trues)
    return scores, 0.0, dict(source=sources, ref=trues, cand=outputs)


def predict_batch(model, sess, eval_batcher, seq_length, word2id, id2word, use_pointer, substr_prefix, verbose=False, **kwargs):
    assert len(word2id) == len(id2word)
    vocab_size = len(word2id)

    outputs = []
    sources = []
    substr_replacement = ' {}'.format(substr_prefix)
    start_time = time.time()
    for batch_index, batch_data in enumerate(eval_batcher.batch(), 1):
        (x_token, x_ids, x_extend, x_mask, oov_size, oovs) = batch_data

        # run encoder
        if verbose:
            print('run encoder')
        fd = model.get_decode_encoder_feed_dict(batch_data, is_predict=True)
        encoder_output = sess.run(model.encoder_output_for_decoder, feed_dict=fd)
        encoder_output = model._split_encoder_output(encoder_output)

        # run decoder step by step
        prev_extend = np.zeros(shape=(x_ids.shape[0], seq_length), dtype=np.int32)
        prev_ids = np.zeros(shape=(x_ids.shape[0], seq_length), dtype=np.int32)
        batch_data.insert(0, prev_ids)
        # not_finish = set(range(y_ids.shape[0]))
        for i in range(seq_length):
            if verbose:
                print('\rrun decoder {}'.format(i), end='')
            batch_data[0] = prev_ids
            fd = model.get_decode_decoder_feed_dict(batch_data=batch_data, split_encoder_output=encoder_output,
                                                    is_predict=True, decoder_seq_length=seq_length)
            preds = sess.run(model.y_pred, feed_dict=fd)
            prev_extend = np.concatenate((preds[:, :i + 1], prev_extend[:, i + 1:]), axis=-1)
            prev_ids = np.copy(prev_extend)
            prev_ids[prev_ids >= vocab_size] = word2id[UNK_TOKEN]

            # 记录已经生成SEP标签的句子，如果所有句子都已经生成SEP标签，则提前结束。
            # delete = set()
            # for index in not_finish:
            #     if word2id[SEP_TOKEN] in preds[index, :i + 1]:
            #         delete.add(index)
            # not_finish -= delete
            # if len(not_finish) == 0:
            #     break

        for i, abs in enumerate(prev_extend.tolist()):
            output = []
            for w in abs:
                if w != word2id[SEP_TOKEN]:
                    if w > 0:
                        output.append(w)
                else:
                    break
            if len(output) == 0:
                output = [word2id[SEP_TOKEN]]
            outputs.append(id2text(ids=output, id2word=id2word, oov=oovs[i]).replace(substr_replacement, ''))
        sources.extend([' '.join(l).replace(substr_replacement, '') for l in x_token.tolist()])
        print('Batch {}, total time: {:.2f}s'.format(batch_index, time.time() - start_time))

    print('Total Eval Time: {:.2f}s'.format(time.time() - start_time))
    return dict(source=sources, cand=outputs)


def predict(FLAGS):
    assert FLAGS.init_checkpoint is not None
    # ************************************************************************

    t = time.time()
    # load parameter from checkpoints file.
    ckpt_path = os.path.join(EXP_DIR, FLAGS.init_checkpoint)
    assert os.path.exists(ckpt_path) and tf.train.latest_checkpoint(ckpt_path) is not None or os.path.isfile(ckpt_path)
    saver = Saver(ckpt_dir=ckpt_path, max_to_keep=CHECKPOINTS_MAX_TO_KEEP)
    config = BertConfig.from_json_file(saver.hyper_parameter_filepath)
    merge_flags_config(FLAGS, config)

    print('\n******************** Hyper parameters: ********************')
    for k, v in config.__dict__.items():
        print('\t{}: {}'.format(k, v))
    print('***********************************************************\n')

    print('Loading data...')
    word2id, id2word = load_vocab(config.vocab_file, do_lower=config.do_lower)
    batcher = get_predict_batcher(src_file=config.test_src,
                                  word2id=word2id, config=config,
                                  batch_size=config.batch_size,
                                  do_lower=config.do_lower,
                                  substr_prefix=config.substr_prefix,
                                  limit=EVAL_SAMPLE_LIMIT)
    print('Time: {:.1f}s'.format(time.time() - t))
    print('Finish loading data...')

    # build model
    model = MultiGPUModel(config=config, num_gpus=config.num_gpus)
    model.build(is_training=False)

    print('GC-ing...')
    gct = time.time()
    gc.collect()
    print('GC Finish! Time: %.1f' % (time.time() - gct))

    # train the model.
    print('Preparing...')
    saver.init_saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.initialize_variables(ckpt_path=config.checkpoint_file)
        sess.run(tf.global_variables_initializer())
        print('Preparing Finish.\n')

        start_time = time.time()
        print('Predicting...')
        texts = predict_batch(model=model, sess=sess, eval_batcher=batcher,
                              seq_length=config.decoder_seq_length, word2id=word2id, id2word=id2word,
                              use_pointer=config.use_pointer, substr_prefix=config.substr_prefix,
                              beam_size=2)
        # saver.summary(loss=loss, scores=scores, prefix='eval', global_step=epoch_index)
        saver.save_summaries(sources=texts['source'], refs=None, cands=texts['cand'],
                             step=saver.ckpt_path.split('/')[-1], suffix=config.mode,
                             folder=os.path.dirname(config.test_src))

        print('Finish. total time:{time:.1f}s'.format(time=time.time() - start_time))



def evaluate(FLAGS):
    assert FLAGS.init_checkpoint is not None
    # ************************************************************************

    t = time.time()
    # load parameter from checkpoints file.
    ckpt_path = os.path.join(EXP_DIR, FLAGS.init_checkpoint)
    assert os.path.exists(ckpt_path) and tf.train.latest_checkpoint(ckpt_path) is not None or os.path.isfile(ckpt_path)
    saver = Saver(ckpt_dir=ckpt_path, max_to_keep=CHECKPOINTS_MAX_TO_KEEP)
    config = BertConfig.from_json_file(saver.hyper_parameter_filepath)
    merge_flags_config(FLAGS, config)

    print('\n******************** Hyper parameters: ********************')
    for k, v in config.__dict__.items():
        print('\t{}: {}'.format(k, v))
    print('***********************************************************\n')

    print('Loading data...')
    word2id, id2word = load_vocab(config.vocab_file, do_lower=config.do_lower)
    batcher = get_batcher(src_file=config.eval_src if config.mode.lower() == 'eval' else config.test_src,
                          dst_file=config.eval_dst if config.mode.lower() == 'eval' else config.test_dst,
                          word2id=word2id, config=config,
                          dst_seq_length=config.decoder_seq_length, batch_size=config.batch_size,
                          do_lower=config.do_lower,
                          substr_prefix=config.substr_prefix,
                          limit=EVAL_SAMPLE_LIMIT)
    print('Time: {:.1f}s'.format(time.time() - t))
    print('Finish loading data...')

    # build model
    model = MultiGPUModel(config=config, num_gpus=config.num_gpus)
    model.build(is_training=False)

    print('GC-ing...')
    gct = time.time()
    gc.collect()
    print('GC Finish! Time: %.1f' % (time.time() - gct))

    # train the model.
    print('Preparing...')
    saver.init_saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.initialize_variables(ckpt_path=config.checkpoint_file)
        sess.run(tf.global_variables_initializer())
        # saver.init_file_writer()
        print('Preparing Finish.\n')

        start_time = time.time()
        print('Evaluating...' if config.mode.lower() == 'eval' else 'Testing...')
        epoch_start = time.time()
        scores, loss, texts = test_batch(model=model, sess=sess, eval_batcher=batcher,
                                         seq_length=config.decoder_seq_length, word2id=word2id, id2word=id2word,
                                         use_pointer=config.use_pointer, substr_prefix=config.substr_prefix,
                                         beam_size=2)
        # saver.summary(loss=loss, scores=scores, prefix='eval', global_step=epoch_index)
        saver.save_summaries(sources=texts['source'], refs=texts['ref'], cands=texts['cand'],
                             step=saver.ckpt_path.split('/')[-1], suffix=config.mode)
        o = ('Rouge-1:{r1:.8}, Rouge-2:{r2:.8}, '
             'Rouge-L:{rl:.8}, time:{time:.1f}s').format(
            total=config.epochs, loss=loss,
            r1=scores['rouge-1']['f'], r2=scores['rouge-2']['f'],
            rl=scores['rouge-l']['f'], time=time.time() - epoch_start)
        print(o)
        print("[{}]{}: {}".format(FLAGS.mode.lower(), ckpt_path, o))

        print('Finish. total time:{time:.1f}s'.format(time=time.time() - start_time))


def train(FLAGS):
    # load and configure hyper-parameters.
    t = time.time()
    if FLAGS.init_checkpoint:
        # load parameter from checkpoints file.
        ckpt = FLAGS.init_checkpoint
    else:
        ckpt = 'checkpoint_{time}'.format(time=time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))


    ckpt_path = os.path.join(EXP_DIR, ckpt)
    ckpt_exist = os.path.exists(ckpt_path) and tf.train.latest_checkpoint(ckpt_path) is not None
    os.makedirs(ckpt_path, exist_ok=True)
    saver = Saver(ckpt_dir=ckpt_path, max_to_keep=CHECKPOINTS_MAX_TO_KEEP)

    if FLAGS.init_checkpoint:
        config = BertConfig.from_json_file(saver.hyper_parameter_filepath)
        # config = BertConfig.from_json_file(FLAGS.bert_config_file)
    else:
        config = BertConfig.from_json_file(FLAGS.bert_config_file)
    merge_flags_config(FLAGS, config)

    if config.train_from_scratch:
        config.gradual_unfreezing = False
        config.discriminative_fine_tuning = False
        config.encoder_trainable_layers = -1
        config.trainable_layers = -1
        config.embedding_trainable = True
        config.pooler_layer_trainable = True
        config.masked_layer_trainable = True
        config.attention_layer_trainable = True

    saver.save_hyper_parameters(config.__dict__)

    print('****** Log content has been redirected to file %s ******' % saver.log_filepath)
    print('****** Please make sure you have save this checkpoint directory! ******')

    print('\n******************** Hyper parameters: ********************')
    for k, v in config.__dict__.items():
        print('\t{}: {}'.format(k, v))
    print('***********************************************************\n')

    print('Loading data...')
    word2id, id2word = load_vocab(config.vocab_file, do_lower=config.do_lower)
    train_batcher = get_batcher(src_file=config.train_src,
                                dst_file=config.train_dst,
                                word2id=word2id, config=config,
                                dst_seq_length=config.decoder_seq_length, batch_size=config.batch_size,
                                do_lower=config.do_lower,
                                substr_prefix=config.substr_prefix,
                                limit=TRAIN_SAMPLE_LIMIT)
    setattr(config, 'steps_per_epoch', train_batcher.iterations)
    print('Time: {:.1f}s'.format(time.time() - t))
    print('Finish loading data...')

    # build model
    # model = Model(config)
    model = MultiGPUModel(config=config, num_gpus=config.num_gpus)
    model.build(is_training=True)

    print('GC-ing...')
    gct = time.time()
    gc.collect()
    print('GC Finish! Time: %.1f' % (time.time() - gct))

    # train the model.
    print('Preparing...')
    start_time = time.time()
    saver.init_saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        print('Start initialize from Saver.')
        if ckpt_exist:
            start_epoch = saver.initialize_variables(ckpt_path=config.checkpoint_file)
        elif FLAGS.train_from_scratch:
            start_epoch = 0
            saver.print_variables()
        else:
            start_epoch = saver.initialize_variables(ckpt_path=config.init_checkpoint, from_bert=True,
                                                     layers_filter=config.hidden_layers_filter)

        bert_learning_rate = config.bert_learning_rate
        other_learning_rate = config.other_learning_rate
        print('Run global_variables_initializer()')
        sess.run(tf.global_variables_initializer())
        saver.init_file_writer(verbose=True)
        print('Preparing Finish.')

        print('Start Training...')
        for epoch_index in range(start_epoch, config.epochs):
            epoch_start = time.time()
            batch_start = time.time()
            # for batch_index, (
            #         y_token, y_ids, y_ids_loss, y_extend, y_mask,
            #         x_token, x_ids, x_extend, x_mask, oov_size, oovs) in enumerate(train_batcher.batch(), 1):
            for batch_index, batch_data in enumerate(train_batcher.batch(), 1):
                if not config.use_pointer:
                    y_ids_loss = batch_data[2]
                    y_ids_loss[y_ids_loss >= len(word2id)] = word2id[UNK_TOKEN]
                fd = model.get_feed_dict(is_training=True, batch_data=batch_data)

                res = sess.run([model.train_op,
                                model.global_step,
                                model.bert_optimizer.learning_rate,
                                model.loss,
                                saver.merged_op],
                               feed_dict=fd,
                               # options=run_options,
                               # run_metadata=run_metadata
                               )

                global_step = res[1]
                lr = res[2]
                loss = res[3]


                # res = sess.run(
                #     [model.models[0].distance,
                #      ... ...
                #      ],
                #     feed_dict=fd
                # )
                #
                # distance = res[0]

                if batch_index % PRINT_STEPS == 0:
                    # loss = sess.run(model.loss, feed_dict=fd)
                    # loss = 0.0
                    saver.summary(loss=loss, prefix='train', global_step=global_step, bert_lr=lr)
                    # saver.file_writer.add_run_metadata(run_metadata, 'step_%d' % global_step)
                    print('batch {i}, Loss:{loss:.8f}, bert_lr={lr:.10f}, time:{time:.1f}s'.format(
                        i=batch_index, loss=loss, time=time.time() - batch_start, lr=lr if lr else 0.0))
                    batch_start = time.time()
                if global_step % CHECK_GLOBAL_STEPS == 0 and global_step != 0 and \
                        (HALVE_BERT_LR and bert_learning_rate >= MIN_LEARNING_RATE or
                         HALVE_OTHER_LR and other_learning_rate >= MIN_LEARNING_RATE):
                    print(('\t{} batches has been trained, scoring validation data set '
                           'for halving the learning rate...').format(CHECK_GLOBAL_STEPS))

            if not config.debug:
                saver.save(sess=sess, step=epoch_index)
            print('Epoch: {i}/{total}, time:{time:.1f}s\n'.format(
                i=epoch_index, total=config.epochs, time=time.time() - epoch_start))
        print('Finish. total time:{time:.1f}s'.format(time=time.time() - start_time))
        saver.close()
    print(ckpt)


def merge_flags_config(flag, config):
    fields = [
        'debug',
        'train_src',
        'train_dst',
        'eval_src',
        'eval_dst',
        'test_src',
        'test_dst',
        'mode',
        'num_gpus',
        'batch_size',
        'learning_rate',
        'bert_learning_rate',
        'other_learning_rate',
        'theta',
        'init_checkpoint',
        'gradual_unfreezing',
        'discriminative_fine_tuning',
        'num_hidden_layers',
        'trainable_layers',
        # 'hidden_layers_filter',     this one needs to be process separately
        'encoder_trainable_layers',
        'embedding_trainable',
        'pooler_layer_trainable',
        'masked_layer_trainable',
        'attention_layer_trainable',
        'pointer_initializer',
        "use_pointer",
        'coverage',
        'trim_attention',
        'align_layers',
        'train_from_scratch',
        'name',
    ]
    for field in fields:
        if getattr(flag, field, None) is not None:
            setattr(config, field, getattr(flag, field))
    include_fileds = [
        'checkpoint_file',
    ]
    for field in include_fileds:
        setattr(config, field, getattr(flag, field))
    if getattr(flag, 'hidden_layers_filter', None) is not None:
        try:
            s = getattr(flag, 'hidden_layers_filter').split(',')
            s = map(int, s)
            setattr(config, 'hidden_layers_filter', tuple(s))
        except:
            traceback.print_exc()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(4399)

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_boolean('debug', False, 'whether save checkpoints or not.')

    flags.DEFINE_string("train_src", None, "")
    flags.DEFINE_string("train_dst", None, "")

    flags.DEFINE_string("eval_src", None, "")
    flags.DEFINE_string("eval_dst", None, "")

    flags.DEFINE_string("test_src", None, "")
    flags.DEFINE_string("test_dst", None, "")
    flags.DEFINE_string('mode', 'train', 'train/eval/test/predict(not support yet)')
    flags.DEFINE_integer("num_gpus", 4, "Number of GPUs")

    flags.DEFINE_integer("batch_size", None, "batch size.")
    flags.DEFINE_float("bert_learning_rate", None, "learning_rate of bert(encoder and decoder).")
    flags.DEFINE_float("other_learning_rate", None, "learning_rate of parts except from bert(encoder and decoder)")
    flags.DEFINE_float("learning_rate", None,
                       "[BACKUP PARAMETERS] learning_rate of parts except from bert(encoder and decoder)")
    flags.DEFINE_float("theta", None, "weight of RL loss function.")
    flags.DEFINE_string("init_checkpoint", None, "initial checkpoint directory.")
    flags.DEFINE_string("checkpoint_file", None,
                        "checkpoint filename in folder ```init_checkpoint``` param.")

    flags.DEFINE_boolean('gradual_unfreezing', None, 'whether use gradual unfreezing or not.')
    flags.DEFINE_boolean('discriminative_fine_tuning', None, 'whether use discriminative fine-tuning or not.')

    flags.DEFINE_integer("num_hidden_layers", None, "number of hidden layers in transformer model.")
    flags.DEFINE_string("hidden_layers_filter", None, "layers of parameters which will be loaded to the model.")

    flags.DEFINE_integer("trainable_layers", None,
                         "number of trainable layers in decoder, -1 means all layers are trainable.")
    flags.DEFINE_integer("encoder_trainable_layers", None,
                         "number of trainable layers in encoder, -1 means all layers are trainable.")
    flags.DEFINE_boolean("embedding_trainable", None, "embedding matrix trainable or not in encoder and decoder.")
    flags.DEFINE_boolean("pooler_layer_trainable", None, "[Deprecated] pooler layer trainable or not in decoder.")
    flags.DEFINE_boolean("masked_layer_trainable", None, "masked layer trainable or not in decoder.")
    flags.DEFINE_boolean("attention_layer_trainable", None, "attention layer trainable or not in decoder.")

    flags.DEFINE_string('pointer_initializer', None,
                        'one of [xavier/normal/truncated], initializer of parameters in Pointer.')
    flags.DEFINE_boolean('use_pointer', None, 'use Pointer Generator Mechanism.')
    flags.DEFINE_boolean('coverage', None, 'use Coverage Mechanism.')
    flags.DEFINE_boolean('trim_attention', None, 'use Trim Relative Self-Attention.')
    flags.DEFINE_boolean('align_layers', None, 'align encoder and decoder layers.')
    flags.DEFINE_string('name', None, 'Name of the experiments')

    flags.DEFINE_string(
         # "bert_config_file", './bert/topic/bert_config.json',
        "bert_config_file", './bert/twitter_bpe/bert_config.json',
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_boolean('train_from_scratch', True,
                         'train from scratch and don\'t use pre-trained BERT parameters.')

    if FLAGS.mode.lower() == 'train':
        try:

            FLAGS = train(FLAGS)
            # FLAGS.checkpoint_file = "best-49"
            # FLAGS.mode = "test"
            # FLAGS.num_gpus = 1
            # evaluate(FLAGS)
            # FLAGS.mode = "eval"
            # evaluate(FLAGS)

        # save_model_graph()
        except:
            traceback.print_exc()
    elif FLAGS.mode.lower() == 'eval' or FLAGS.mode.lower() == 'test':
        evaluate(FLAGS)
    elif FLAGS.mode.lower() == 'predict' or FLAGS.mode.lower() == 'decode':
        predict(FLAGS)
    else:
        eprint('[ERROR] Mode parameter should be train/eval/test/predict.')
