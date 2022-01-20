import re
import sys
import time

import numpy as np


sys.path.append("..")
sys.path.append(".")

from constants import SAMPLE_LIMIT, PAD_TOKEN
from constants import SEP_TOKEN, UNK_TOKEN, CLS_TOKEN, SENTEN_TOKEN
from utils import tokenization



def load_vocab(vocab_file, do_lower=False):
    word2id, id2word = dict(), dict()
    with open(vocab_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            w = line.strip('\n')
            if do_lower:
                if not w[0] == '[' or not w[-1] == ']':
                    w = w.lower()
            word2id[w] = i
            id2word[i] = w

    print(len(word2id))
    print(len(id2word))
    assert len(word2id) == len(id2word)
    for i in range(len(word2id)):
        assert word2id[id2word[i]] == i
    return word2id, id2word


def load_text(file, do_lower, vocab, substr_prefix='##'):
    print('Loading file: {}'.format(file))

    tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=do_lower, substr_prefix=substr_prefix)
    i = -1
    with open(file, 'r', encoding='utf8') as f:
        tokens = []
        for i, l in enumerate(f):
            l = tokenizer.tokenize(l)
            tokens.append(l)

            if i % 100 == 0:
                print('\r{}'.format(i), end='')
    print('\r{}/{}'.format(i + 1, i + 1))
    return tokens


def load_src(src_file, config, do_lower, vocab=None, substr_prefix='##', blank_reg=re.compile('[ \t\n]'),
             limit=SAMPLE_LIMIT):
    print('Loading src file: {}'.format(src_file))

    # segment_len = 8
    # segment_num = 8
    seq_length = config.encoder_seq_length if config is not None else 100
    segment_len = config.segment_length if config is not None else 10
    segment_num = config.segment_number if config is not None else 10

    # segment_len = 15
    # segment_num = 5

    if src_file.endswith('.token'):
        tokenize = lambda x: x.strip().split(' ')
    else:
        tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=do_lower, substr_prefix=substr_prefix)
        tokenize = lambda x: tokenizer.tokenize(x)

    with open(src_file, 'r', encoding='utf8') as f:
        #     tokens = [list(blank_reg.sub('', line)) for line in f]
        tokens = []
        mask = []
        ids = []
        ids_extend = []
        oovs = []
        oov_size = []
        for i, l in enumerate(f):
            l = tokenize(l)
            if vocab:

                CLS_ID = vocab[CLS_TOKEN]
                SEP_ID = vocab[SEP_TOKEN]
                PAD_ID = vocab[PAD_TOKEN]
                UNK_ID = vocab[UNK_TOKEN]
                SENTEN_ID = vocab[SENTEN_TOKEN]

                oov = []
                tmp_token = []
                tmp_extend = []
                tmp = []

                max_word_number = seq_length - segment_num - 1  # subtract segment number(15) * [CLS], [SENTEN] and [SEP]
                trimmed_sentence = l[:max_word_number-1]
                # add [SEP] token as the end of sentence
                trimmed_sentence.append(SEP_TOKEN)
                trimmed_sentence += [PAD_TOKEN] * (max_word_number - len(trimmed_sentence))  # to 225

                # assert len(trimmed_sentence) == 225

                # print(max_word_number)

                for w_index, w in enumerate(trimmed_sentence):

                    # add [CLS] as the begin of each segment(segment length is 15)
                    if w_index % segment_len == 0: # todo
                        tmp_token.append(CLS_TOKEN)
                        tmp.append(CLS_ID)
                        tmp_extend.append(CLS_ID)

                    # TODO: Need some token at the end of each segment ?

                    # solve w, oov, unk etc...
                    tmp_token.append(w)
                    if w in vocab:
                        tmp.append(vocab[w])
                        tmp_extend.append(vocab[w])
                    elif w in oov:
                        tmp.append(vocab['[UNK]'])
                        tmp_extend.append(len(vocab) + oov.index(w))
                    else:
                        oov.append(w)
                        tmp.append(vocab['[UNK]'])
                        tmp_extend.append(len(vocab) + oov.index(w))

                # add [SENTEN] token as the begin of sentence
                tmp_token = [SENTEN_TOKEN] + tmp_token
                tmp_extend = [SENTEN_ID] + tmp_extend
                tmp = [SENTEN_ID] + tmp

                first_mask_pos = len(tmp)
                for w_index, w in enumerate(tmp):
                    if w == PAD_ID:
                        first_mask_pos = w_index  # get the first pad position
                        break
                # if no pad in tmp, tmp mask is all [1]
                # else, get the first pad position
                #   the position is also indicate number of
                #   no pad token in the sentence
                tmp_mask = [1] * first_mask_pos + [0] * (len(tmp) - first_mask_pos)

                # assert len(tmp_token) == 241
                # assert len(tmp_extend) == 241
                # assert len(tmp) == 241
                # assert len(tmp_mask) == 241

                # return
                mask.append(tmp_mask)
                tokens.append(tmp_token)
                ids_extend.append(tmp_extend[:seq_length])
                ids.append(tmp[:seq_length])
                oovs.append(oov)
                oov_size.append(len(oov))

                # old version backup
                # oov = []
                # tmp_token = [CLS_TOKEN]
                # tmp_extend = [vocab[CLS_TOKEN]]
                # tmp = [vocab[CLS_TOKEN]]
                # for w in l[:seq_length - 2]:
                #     tmp_token.append(w)
                #     if w in vocab:
                #         tmp.append(vocab[w])
                #         tmp_extend.append(vocab[w])
                #     elif w in oov:
                #         tmp.append(vocab[UNK_TOKEN])
                #         tmp_extend.append(len(vocab) + oov.index(w))
                #     else:
                #         oov.append(w)
                #         tmp.append(vocab[UNK_TOKEN])
                #         tmp_extend.append(len(vocab) + oov.index(w))
                # tmp_token.append(SEP_TOKEN)
                # tmp_extend.append(vocab[SEP_TOKEN])
                # tmp.append(vocab[SEP_TOKEN])
                # mask.append(([1] * len(tmp_extend) + [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend)))[:seq_length])
                #
                # tmp_extend += [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend))
                # tmp += [vocab[PAD_TOKEN]] * (seq_length - len(tmp))
                #
                # tokens.append(tmp_token)
                # ids_extend.append(tmp_extend[:seq_length])
                # ids.append(tmp[:seq_length])
                # oovs.append(oov)
                # oov_size.append(len(oov))
            else:
                mask.append(([1] * len(l) + [0] * (seq_length - len(l)))[:seq_length])

            if i % 1000 == 0:
                print('\r{}/{}'.format(i, limit), end='')
            if limit is not None and len(ids) >= limit:
                break
    print('\r{}/{}'.format(i + 1, i + 1))
    if vocab:
        return np.array(tokens), np.array(ids), np.array(ids_extend), np.array(mask), np.array(oov_size), np.array(oovs)
    else:
        return np.array(tokens), np.array(mask)


def load_dst(dst_file, seq_length, do_lower, vocab, src_oovs=None, src_ids=None, substr_prefix='##',
             blank_reg=re.compile('[ \t\n]'),
             limit=SAMPLE_LIMIT):
    print('Loading dst file: {}'.format(dst_file))
    assert vocab is not None
    if src_oovs is not None or src_ids is not None:
        assert src_oovs is not None
        assert src_ids is not None
        assert len(src_oovs) == len(src_ids)

    vocab_size = len(vocab)

    if dst_file.endswith('.token'):
        tokenize = lambda x: x.strip().split(' ')
    else:
        tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=do_lower, substr_prefix=substr_prefix)
        tokenize = lambda x: tokenizer.tokenize(x)

    with open(dst_file, 'r', encoding='utf8') as f:
        tokens = []
        mask = []
        # token_id, 用于输入，模型中会在最前面插入CLS embedding，用于预测第一个词。不需要结尾的SEP标签，集外词标记为UNK
        ids = []
        # extended token_id, 在token_id的基础上，集外词由额外的ID表示。
        ids_extend = []
        # extended token_id with SEP label, 在extended token_id的基础上，末尾加上了SEP标签。应当作为学习目标并用于计算loss
        ids_ext_sep = []
        for i, l in enumerate(f):
            if limit is not None and i >= limit:
                continue
            # print(l)
            l = tokenize(l)
            tmp_l = []
            i = 0
            while i < len(l):
                if i < len(l)-2 and ''.join([l[i], l[i+1], l[i+2]]) == '[SEP]':
                    tmp_l.append('#')
                    i += 3
                else:
                    tmp_l.append(l[i])
                    i+= 1

            l = tmp_l
            tmp_token = []
            tmp_extend = []
            tmp_id = []
            for w in l[:seq_length - 1]:
                if w != UNK_TOKEN and w in vocab:
                    tmp_extend.append(vocab[w])
                elif w in src_oovs[i]:
                    tmp_extend.append(vocab_size + src_oovs[i].index(w))
                else:
                    tmp_extend.append(vocab[UNK_TOKEN])
                tmp_id.append(vocab[w] if w in vocab else vocab[UNK_TOKEN])
                tmp_token.append(w)

            tmp_token.append(SEP_TOKEN)
            tokens.append(tmp_token)
            # tmp_id don't need to add CLS_ID, because we use `CLS embedding` of encoder's output.
            ids_ext_sep.append(
                tmp_extend + [vocab[SEP_TOKEN]] + [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend) - 1))

            mask.append(([1] * (len(tmp_extend) + 1) + [0] * (seq_length - len(tmp_extend) - 1))[:seq_length])

            tmp_extend += [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend))
            ids_extend.append(tmp_extend[:seq_length])

            tmp_id += [vocab[PAD_TOKEN]] * (seq_length - len(tmp_id))
            ids.append(tmp_id[:seq_length])

        if i % 1000 == 0:
            print('\r{}/{}'.format(i, limit), end='')
    print('\r{}/{}'.format(i + 1, i + 1))
    return np.array(tokens[:limit]), np.array(ids), np.array(ids_extend), np.array(ids_ext_sep), np.array(mask)


def count(src_file, dst_file, vocab, substr_prefix='##'):
    start_time = time.time()
    tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=True, substr_prefix=substr_prefix)
    print('counting src...')
    with open(src_file, 'r', encoding='utf8') as f:
        #     tokens = [list(blank_reg.sub('', line)) for line in f]
        src_lengths = []
        for i, l in enumerate(f):
            l = tokenizer.tokenize(l)
            src_lengths.append(len(l))
            if i % 2000 == 0:
                print('\r{}, time: {:.2f}s'.format(i, time.time() - start_time), end='')
        print('\r{}, time: {:.2f}s'.format(i + 1, time.time() - start_time))
    src_lengths.sort()
    print('\ncounting dst...')
    with open(dst_file, 'r', encoding='utf8') as f:
        dst_lengths = []
        for i, l in enumerate(f):
            l = tokenizer.tokenize(l)
            dst_lengths.append(len(l))
            if i % 2000 == 0:
                print('\r{}, time: {:.2f}s'.format(i, time.time() - start_time), end='')
        print('\r{}, time: {:.2f}s'.format(i + 1, time.time() - start_time))
    dst_lengths.sort()
    print('Time of Loading Training Data: {:.2f}s\n'.format(time.time() - start_time))
    print('\nSRC:\nMax:{max}\nMin:{min}\nAverage:{avg}\n95%:{r95}\n98%:{r98}'.format(
        max=max(src_lengths),
        min=min(src_lengths),
        avg=sum(src_lengths) / len(src_lengths),
        r95=src_lengths[round(len(src_lengths) * 0.95)],
        r98=src_lengths[round(len(src_lengths) * 0.98)],
    ))
    print()
    print('DST:\nMax:{max}\nMin:{min}\nAverage:{avg}\n95%:{r95}\n98%:{r98}'.format(
        max=max(dst_lengths),
        min=min(dst_lengths),
        avg=sum(dst_lengths) / len(dst_lengths),
        r95=dst_lengths[round(len(dst_lengths) * 0.95)],
        r98=dst_lengths[round(len(dst_lengths) * 0.98)],
    ))
    return src_lengths, dst_lengths


def id2text(ids, id2word, oov, vocab_size=None):
    if vocab_size is None:
        vocab_size = len(id2word)
    text = []
    if type(oov) != list:
        oov = oov.tolist()
    for i in ids:
        if i in id2word:
            text.append(id2word[i])
        else:
            text.append(oov[i - vocab_size])
    return ' '.join(text)


def ids2text(ids, id2word, oovs):
    vocab_size = len(id2word)
    texts = []
    for id_, oov in zip(ids, oovs):
        texts.append(id2text(ids=id_, id2word=id2word, oov=oov, vocab_size=vocab_size))
    return texts


if __name__ == '__main__':
    # load vocab ...
    train_src = "../data/twitter_sample/twitter_train.src"
    encoder_seq_length = 81
    train_dst = "../data/twitter_sample/twitter_train.dst"
    decoder_seq_length = 9
    vocab, _ = load_vocab("../bert/twitter/vocab.txt")

    # train_src = "../data/topic_sample/sample.src.token"
    # encoder_seq_length = 81
    # train_dst = "../data/topic_sample/sample.dst.token"
    # decoder_seq_length = 9
    # vocab, _ = load_vocab("../bert/topic_ltp/vocab.txt")

    print('finish load vocab...')

    # test load src
    src, src_ids, _, mask, _, src_oovs = load_src(src_file=train_src,
                                                  config=None,
                                                  do_lower=False,
                                                  vocab=vocab)


    print('test load src... ')
    print(src[0])
    print(mask[0])
    print(src_ids[0])
    print(len(src[0]))
    # exit(0)

    dst, x, y, z, mask = load_dst(dst_file=train_dst,
                                  seq_length=decoder_seq_length,
                                  do_lower=False,
                                  vocab=vocab,
                                  src_oovs=src_oovs,
                                  src_ids=src_ids)
    print('test load dst... ')
    print(dst[0])
    print(mask[0])
    print(len(mask[0]))
    # print(x[0])
    # print(y[0])
    # print(z[0])


    print('finish!')
