import math

import numpy as np
import tensorflow as tf

from utils import get_shape_list


def my_self_relative_attention(q, k, v, mask, relative_size, heads=12):
    """
    Relative self attention implemented by tangbin.
    Created at 2019/6/15.
    :param q: [Batch_size, seq_length, heads*dim_per_heads]
    :param k: [Batch_size, seq_length, heads*dim_per_heads]
    :param v: [Batch_size, seq_length, heads*dim_per_heads]
    :param mask: [Batch_size, seq_length]
    :param relative_size: Actual size: 2 * relative_size + 1
    :param batch_size: Batch_size
    :param seq_length:
    :param hidden_dim:
    :param heads:
    :return:
    """
    print('[INPUT] q.shape: {}, k.shape: {}, v.shape: {}'.format(q.shape, k.shape, v.shape))
    # assert q.shape == k.shape == v.shape
    batch_size, seq_length, hidden_dim = get_shape_list(q)
    dim_per_head = hidden_dim // heads

    q = tf.reshape(q, shape=(batch_size, seq_length, heads, dim_per_head))
    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.reshape(k, shape=(batch_size, seq_length, heads, dim_per_head))
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.reshape(v, shape=(batch_size, seq_length, heads, dim_per_head))
    v = tf.transpose(v, [0, 2, 1, 3])
    print('[TRANSPOSED] q.shape: {}, k.shape: {}, v.shape: {}'.format(q.shape, k.shape, v.shape))
    # shapes of q/k/v: [batch_size, heads, seq_length, dim_per_head]
    contexts = []
    for i, data in enumerate(tf.unstack(q, axis=2)):
        # data: [batch_size, 1, heads, dim_per_head]
        data = tf.expand_dims(data, axis=2)
        print('[my_self_attention] i:{}, data:{}'.format(i, data.shape))
        left = max(0, i - relative_size)
        right = min(seq_length - 1, i + relative_size)
        print('[Relative] left:{}, right:{}, size:{} '.format(left, right, right - left + 1))
        # [batch_size, number of heads, 2 * relative_size + 1, hidden_dim per head]
        enc_k = tf.slice(k, [0, 0, left, 0], [batch_size, heads, right - left + 1, dim_per_head])
        enc_v = tf.slice(v, [0, 0, left, 0], [batch_size, heads, right - left + 1, dim_per_head])
        print('enc_k/enc_v', enc_k.shape)
        # [batch_size, heads, 1, 2 * relative_size + 1]
        # scores = tf.matmul(data, enc_k, transpose_b=True)
        scores = tf.einsum('hijl,hikl->hijk', data, enc_k)
        scores = tf.multiply(scores, 1.0 / math.sqrt(dim_per_head))
        print('scores:', scores.shape)

        print('mask:', mask.shape)
        msk = tf.slice(mask, [0, i, left], [batch_size, i + 1, right - left + 1])
        msk = tf.reshape(msk, [batch_size, 1, 1, right - left + 1])
        adder = (1.0 - tf.cast(msk, tf.float32)) * -1e6
        scores += adder

        proba = tf.nn.softmax(scores, axis=-1)
        print('proba/scores', proba.shape)
        ct = tf.einsum('hijk,hikl->hil', proba, enc_v)
        print('ct', ct.shape)
        contexts.append(ct)
    context = tf.stack(contexts, axis=1)
    context = tf.reshape(context, shape=[batch_size, seq_length, hidden_dim])
    return context


_pad_mask_dict = {}


def get_pad_mask(relative_size, seq_length, heads):
    key = (relative_size, seq_length)
    if key in _pad_mask_dict:
        pad_mask = _pad_mask_dict[key]
    else:
        pad_mask = [[1] * max(relative_size - i, 0) + [0] * (
                2 * relative_size + 1 - max(relative_size - i, 0) - max(relative_size + i - seq_length, 0)) + [1] * max(
            relative_size + i - seq_length, 0) for i in range(seq_length)]
        pad_mask = tf.constant(pad_mask, dtype=tf.float32)
        pad_mask = tf.tile(tf.expand_dims(pad_mask, axis=0), [heads, 1, 1])
        pad_mask = tf.expand_dims(pad_mask, axis=0)
        # pad_mask = np.array(pad_mask)
        _pad_mask_dict[key] = pad_mask
    # print('pad_mask.shape:', pad_mask.shape)
    return pad_mask


def my_self_relative_attention_v2(q, k, v, relative_size, mask_matrix, heads=12, session=None, **kwargs):
    """
    This is an implementation of my self relative attention v2.
    It should be used in encoder.
    This function doesn't support masked self attention.
    :param q: shape = [batch_size, seq_length, hidden_dim]
    :param k: shape = [batch_size, seq_length, hidden_dim]
    :param v: shape = [batch_size, seq_length, hidden_dim]
    :param relative_size:
    :param mask_matrix: shape = [batch_size, seq_length]
    :param heads: num of heads.
    :param session: if session is not None, calc and print the value of the nodes using the session.
    :return:
    """
    # print('q.shape:{}\nk.shape:{}\nv.shape:{}'.format(q.shape, k.shape, v.shape))
    batch_size, seq_length, hidden_dim = get_shape_list(q)
    # if batch_size is None:
    #     batch_size = b
    # elif type(batch_size) == int and type(b) == int:
    #     assert b == batch_size
    dim_per_head = hidden_dim // heads

    if mask_matrix is not None and mask_matrix.dtype != tf.float32:
        mask_matrix = tf.cast(mask_matrix, dtype=tf.float32)

    # if heads > 1:
    q = tf.reshape(q, shape=(batch_size, seq_length, heads, dim_per_head))
    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.reshape(k, shape=(batch_size, seq_length, heads, dim_per_head))
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.reshape(v, shape=(batch_size, seq_length, heads, dim_per_head))
    v = tf.transpose(v, [0, 2, 1, 3])
    # print('[TRANSPOSED] q.shape: {}\n[TRANSPOSED] k.shape: {}\n[TRANSPOSED] v.shape: {}'.format(q.shape, k.shape, v.shape))

    # get padding k
    # q/k/v.shape = [batch_size, heads, seq_length, dim_per_head]
    pad_for_k = tf.zeros(shape=(relative_size, batch_size, heads, dim_per_head))
    transpose_k = tf.transpose(k, [2, 0, 1, 3])  # [seq_length, batch_size, heads, hidden_dim]
    pad_k = tf.concat([pad_for_k, transpose_k, pad_for_k], axis=0)
    # print('pad_k.shape:', pad_k.shape)
    if session:
        print('pad_k:{}'.format(session.run(pad_k)))
    
    # get padding v
    transpose_v = tf.transpose(v, [2, 0, 1, 3])  # [seq_length, batch_size, heads, hidden_dim]
    pad_v = tf.concat([pad_for_k, transpose_v, pad_for_k], axis=0)
    # print('pad_v.shape:', pad_v.shape)
    if session:
        print('pad_v:{}'.format(session.run(pad_v)))

    # gather k
    indices = [list(range(i, 2 * relative_size + 1 + i)) for i in range(seq_length)]
    gather_k = tf.gather(pad_k, indices)  # [seq_length, 2c+1, batch_size, heads, dim_per_head]
    gather_v = tf.gather(pad_v, indices)  # [seq_length, 2c+1, batch_size, heads, dim_per_head]
    # transposed version:
    # gather_k = tf.transpose(gather_k, [2, 3, 0, 1, 4])  # [batch_size, heads, seq_length, 2c+1, dim_per_head]
    # print('gather_k.shape:', gather_k.shape)
    # score = tf.einsum('hijkp,hijp->hijk', gather_k, q)  # [batch_size, heads, seq_length, 2c+1]
    # un-transposed version:
    # print('gather_k.shape:', gather_k.shape)  # [seq_length, 2c+1, batch_size, heads, dim_per_head]
    score = tf.einsum('jkhip,hijp->hijk', gather_k, q)  # [batch_size, heads, seq_length, 2c+1]
    if session:
        print('indices:{}'.format(indices))
        print('gather_k:{}'.format(session.run(gather_k)))
        print('score:{}'.format(session.run(score)))

    # get padding mask and gather it
    if mask_matrix is not None:
        transpose_mask = tf.transpose(mask_matrix, [1, 0])
        pad_for_mask = tf.zeros(shape=(relative_size, batch_size), dtype=tf.float32)
        pad_mask = tf.concat([pad_for_mask, transpose_mask, pad_for_mask], axis=0)
        gather_mask = tf.gather(pad_mask, indices=indices)
        final_mask = 1.0 - tf.expand_dims(tf.transpose(gather_mask, [2, 0, 1]), 1)
        if session:
            print('final_mask:{}'.format(session.run(final_mask)))
    else:
        # pad_mask is useless, because input_mask is more strict than pad mask.
        pad_mask = get_pad_mask(relative_size=relative_size, seq_length=seq_length, heads=heads)
        final_mask = pad_mask

    # calculate attention and context
    # print('score.shape:', score.shape)
    masked_score = score + final_mask * (-1e6)
    if session:
        print('masked_score:{}'.format(session.run(masked_score)))

    proba = tf.nn.softmax(masked_score, axis=-1)
    # print('proba.shape:', proba.shape)
    if session:
        print('proba:{}'.format(session.run(proba)))
    # transposed version:
    # context = tf.einsum('hijkp,hijk->hjip', gather_k, proba)
    # un-transposed version:
    context = tf.einsum('jkhip,hijk->hjip', gather_v, proba)
    context = tf.reshape(context, shape=[batch_size, seq_length, hidden_dim])
    # print('context.shape:', context.shape)
    if session:
        print('context:{}'.format(session.run(context)))
    return context


def test_my_self_relative_attention_v2():
    np.random.seed(233)
    tf.set_random_seed(233)
    s = tf.Session()

    batch_size = 3
    seq_length = 7
    hidden_dim = 2
    heads = 1
    relative_size = 1
    a = np.random.random((batch_size, seq_length, hidden_dim))
    a = tf.constant(a, dtype=tf.float32)
    mask = tf.constant(np.concatenate((np.ones((batch_size, 4)), np.zeros((batch_size, 3))), axis=-1), dtype=np.float32)
    print(s.run(mask))
    ct = my_self_relative_attention_v2(a, a, a, heads=heads, mask_matrix=mask, relative_size=relative_size, session=s)
    print(ct.shape)
    print(s.run(tf.reduce_sum(ct)))


def test():
    test_my_self_relative_attention_v2()


if __name__ == '__main__':
    test()
