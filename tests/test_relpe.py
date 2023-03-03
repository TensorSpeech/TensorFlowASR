import tensorflow as tf

from tensorflow_asr.models.layers import positional_encoding
from tensorflow_asr.utils import math_util, plot_util


def test():
    batch_size, max_length, dmodel = 1, 500, 144
    pe = positional_encoding.compute_sinusoid_position_encoding(batch_size, max_length, dmodel)
    mask = tf.sequence_mask([500], max_length)[..., None]
    pe = math_util.apply_mask(pe, mask=mask)
    pe = pe[0]
    pe = pe.numpy().T
    print(pe.shape)
    plot_util.plotmesh(pe, title="sinusoid position encoding")
