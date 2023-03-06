import tensorflow as tf

from tensorflow_asr.models.layers import positional_encoding
from tensorflow_asr.utils import math_util, plot_util


def test():
    batch_size, input_length, max_length, dmodel = 1, 450, 500, 144
    pe = positional_encoding.compute_sinusoid_position_encoding(max_length, dmodel)
    pe = pe.numpy().T
    print(pe.shape)
    plot_util.plotmesh(pe, title="sinusoid position encoding")
