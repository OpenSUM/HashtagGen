import numpy as np

from utils import BertConfig
from .data_loader import load_src, load_dst, load_vocab, load_text


class Batcher:
    def __init__(self, y_token, y_ids, y_ids_ext_sep, y_ids_extend, y_mask, x_token, x_ids, x_ids_extend,
                 # x_output,
                 x_mask, oov_size, oovs, batch_size):
        assert y_token.shape[0] == y_ids.shape[0] == y_mask.shape[0] == x_ids.shape[0] == \
               x_mask.shape[0] == oov_size.shape[0] == oovs.shape[0] == x_token.shape[0]
        # x_output.shape[0] == \

        np.random.seed(7890)
        self.indices = np.random.permutation(range(y_ids.shape[0]))
        self.iterations = (y_ids.shape[0] - 1) / batch_size + 1
        self.total_samples = y_ids.shape[0]

        self.y_token = y_token[self.indices]
        self.y_ids = y_ids[self.indices]
        self.y_ids_ext_sep = y_ids_ext_sep[self.indices]
        self.y_ids_extend = y_ids_extend[self.indices]
        self.y_mask = y_mask[self.indices]
        self.x_token = x_token[self.indices]
        self.x_ids = x_ids[self.indices]
        self.x_ids_extend = x_ids_extend[self.indices]
        # self.x_output = x_output[self.indices]
        self.x_mask = x_mask[self.indices]
        self.oov_size = oov_size[self.indices]
        self.oovs = oovs[self.indices]
        self.batch_size = batch_size

    def shuffle(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.indices = np.random.permutation(range(self.y_ids.shape[0]))

        self.y_token = self.y_token[self.indices]
        self.y_ids = self.y_ids[self.indices]
        self.y_ids_ext_sep = self.y_ids_ext_sep[self.indices]
        self.y_ids_extend = self.y_ids_extend[self.indices]
        self.y_mask = self.y_mask[self.indices]
        self.x_token = self.x_token[self.indices]
        self.x_ids = self.x_ids[self.indices]
        self.x_ids_extend = self.x_ids_extend[self.indices]
        # self.x_output = self.x_output[self.indices]
        self.x_mask = self.x_mask[self.indices]
        self.oov_size = self.oov_size[self.indices]
        self.oovs = self.oovs[self.indices]

    def batch(self, batch_size=None):
        self.shuffle()
        i = 0
        if batch_size is None:
            batch_size = self.batch_size
        size = self.y_ids.shape[0]
        while batch_size * (i + 1) < size:
            yield [self.y_token[i * batch_size:(i + 1) * batch_size],
                   self.y_ids[i * batch_size:(i + 1) * batch_size],
                   self.y_ids_ext_sep[i * batch_size:(i + 1) * batch_size],
                   self.y_ids_extend[i * batch_size:(i + 1) * batch_size],
                   self.y_mask[i * batch_size:(i + 1) * batch_size],
                   self.x_token[i * batch_size:(i + 1) * batch_size],
                   self.x_ids[i * batch_size:(i + 1) * batch_size],
                   self.x_ids_extend[i * batch_size:(i + 1) * batch_size],
                   # self.x_output[i * batch_size:(i + 1) * batch_size],
                   self.x_mask[i * batch_size:(i + 1) * batch_size],
                   self.oov_size[i * batch_size:(i + 1) * batch_size],
                   self.oovs[i * batch_size:(i + 1) * batch_size], ]
            i += 1
        yield [self.y_token[i * batch_size:],
               self.y_ids[i * batch_size:],
               self.y_ids_ext_sep[i * batch_size:],
               self.y_ids_extend[i * batch_size:],
               self.y_mask[i * batch_size:],
               self.x_token[i * batch_size:],
               self.x_ids[i * batch_size:],
               self.x_ids_extend[i * batch_size:],
               # self.x_output[i * batch_size:],
               self.x_mask[i * batch_size:],
               self.oov_size[i * batch_size:],
               self.oovs[i * batch_size:], ]


class PredictBatcher:
    def __init__(self, x_token, x_ids, x_ids_extend,
                 # x_output,
                 x_mask, oov_size, oovs, batch_size):
        assert x_ids.shape[0] == x_mask.shape[0] == oov_size.shape[0] == oovs.shape[0] == x_token.shape[0]
        # x_output.shape[0] == \

        np.random.seed(7890)
        self.iterations = (x_ids.shape[0] - 1) / batch_size + 1
        self.total_samples = x_ids.shape[0]

        self.x_token = x_token
        self.x_ids = x_ids
        self.x_ids_extend = x_ids_extend
        # self.x_output = x_output
        self.x_mask = x_mask
        self.oov_size = oov_size
        self.oovs = oovs
        self.batch_size = batch_size

    def shuffle(self, seed=None):
        pass

    def batch(self, batch_size=None):
        self.shuffle()
        i = 0
        if batch_size is None:
            batch_size = self.batch_size
        size = self.x_ids.shape[0]
        while batch_size * (i + 1) < size:
            yield [self.x_token[i * batch_size:(i + 1) * batch_size],
                   self.x_ids[i * batch_size:(i + 1) * batch_size],
                   self.x_ids_extend[i * batch_size:(i + 1) * batch_size],
                   # self.x_output[i * batch_size:(i + 1) * batch_size],
                   self.x_mask[i * batch_size:(i + 1) * batch_size],
                   self.oov_size[i * batch_size:(i + 1) * batch_size],
                   self.oovs[i * batch_size:(i + 1) * batch_size], ]
            i += 1
        yield [self.x_token[i * batch_size:],
               self.x_ids[i * batch_size:],
               self.x_ids_extend[i * batch_size:],
               # self.x_output[i * batch_size:],
               self.x_mask[i * batch_size:],
               self.oov_size[i * batch_size:],
               self.oovs[i * batch_size:], ]


def get_batcher(src_file, dst_file, word2id, config, dst_seq_length, batch_size, do_lower, substr_prefix='##',
                limit=None):
    src_tokens, src_ids, src_ids_extend, src_mask, src_oov_size, src_oovs = load_src(src_file=src_file,
                                                                                     config=config,
                                                                                     do_lower=do_lower,
                                                                                     vocab=word2id,
                                                                                     substr_prefix=substr_prefix,
                                                                                     limit=limit)
    dst_tokens, dst_ids, dst_ids_extend, dst_ids_ext_sep, dst_mask = load_dst(dst_file=dst_file,
                                                                              seq_length=dst_seq_length,
                                                                              do_lower=do_lower,
                                                                              vocab=word2id,
                                                                              src_oovs=src_oovs,
                                                                              src_ids=src_ids,
                                                                              substr_prefix=substr_prefix,
                                                                              limit=limit)

    print('Example tokens:')
    for i in range(min(1, len(src_tokens), len(dst_tokens))):
        print('src: {}\ndst: {}\n'.format(' '.join(src_tokens[i]), ' '.join(dst_tokens[i])))

    batcher = Batcher(y_token=dst_tokens,
                      y_ids=dst_ids,
                      y_ids_ext_sep=dst_ids_ext_sep,
                      y_ids_extend=dst_ids_extend,
                      y_mask=dst_mask,
                      x_token=src_tokens,
                      x_ids=src_ids,
                      x_ids_extend=src_ids_extend,
                      # x_output=src_embeddings,
                      x_mask=src_mask,
                      oov_size=src_oov_size,
                      oovs=src_oovs,
                      batch_size=batch_size)
    return batcher


def get_predict_batcher(src_file, word2id, config, batch_size, do_lower, substr_prefix='##', limit=None):
    src_tokens, src_ids, src_ids_extend, src_mask, src_oov_size, src_oovs = load_src(src_file=src_file,
                                                                                     config=config,
                                                                                     do_lower=do_lower,
                                                                                     vocab=word2id,
                                                                                     substr_prefix=substr_prefix,
                                                                                     limit=limit)

    print('Example tokens:')
    for i in range(min(1, len(src_tokens))):
        print('src: {}\n'.format(' '.join(src_tokens[i])))

    batcher = PredictBatcher(x_token=src_tokens,
                             x_ids=src_ids,
                             x_ids_extend=src_ids_extend,
                             # x_output=src_embeddings,
                             x_mask=src_mask,
                             oov_size=src_oov_size,
                             oovs=src_oovs,
                             batch_size=batch_size)
    return batcher


def prepare():
    config = BertConfig.from_json_file('./bert/lcsts/bert_config.json')
    word2id, id2word = load_vocab(config.vocab_file)
    for fp in [config.train_src, config.train_dst, config.eval_src, config.eval_dst, config.test_src, config.test_dst]:
        tokens = load_text(file=fp, do_lower=config.do_lower, vocab=word2id, substr_prefix=config.substr_prefix)
        sentences = [' '.join(l) for l in tokens]
        with open(fp + '.token', 'w', encoding='utf8') as f:
            print('Example tokens:')
            for i, s in enumerate(sentences):
                if i < 3:
                    print(s)
                f.write(s)
                f.write('\n')


if __name__ == '__main__':
    prepare()
