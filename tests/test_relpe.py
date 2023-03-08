import tensorflow as tf

from tensorflow_asr.models.layers import positional_encoding
from tensorflow_asr.models.layers.multihead_attention import rel_left_shift
from tensorflow_asr.utils import math_util, plot_util


def test():
    batch_size, input_length, max_length, dmodel = 1, 450, 500, 144
    pe = positional_encoding.compute_sinusoid_position_encoding(max_length, dmodel, input_length)
    pe = pe.numpy().T
    print(pe.shape)
    plot_util.plotmesh(pe, title="sinusoid position encoding")


def test_relshift():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)
    a = a[None, ...]
    a = a[None, ...]
    b = rel_left_shift(a)
    b = tf.squeeze(b, 0)
    b = tf.squeeze(b, 0)
    print(b)
