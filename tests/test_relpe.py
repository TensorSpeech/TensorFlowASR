import tensorflow as tf

from tensorflow_asr.models.layers.multihead_attention import rel_left_shift
from tensorflow_asr.models.layers.positional_encoding import RelativePositionalEncoding
from tensorflow_asr.utils import plot_util


def test():
    batch_size, input_length, max_length, dmodel = 1, 300, 500, 144
    layer = RelativePositionalEncoding(interleave=True, reverse=True, memory_length=input_length)
    _, pe = layer(tf.random.normal([batch_size, max_length, dmodel]), training=False)
    pe = tf.transpose(pe[0], perm=[1, 0])
    shift = rel_left_shift(tf.reshape(pe, [1, 1] + pe.shape.as_list()))
    pe = pe.numpy()
    print(pe.shape)
    plot_util.plotmesh(pe, title="sinusoid position encoding", invert_yaxis=False)
    plot_util.plotmesh(shift[0][0], title="relshift")


def test_relshift():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)
    a = a[None, ...]
    a = a[None, ...]
    b = rel_left_shift(a)
    b = tf.squeeze(b, 0)
    b = tf.squeeze(b, 0)
    print(b)
