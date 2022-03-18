from tensorflow_asr.models.layers import positional_encoding
from tensorflow_asr.utils import plot_util


def test():
    batch_size, max_length, dmodel = 1, 512, 144
    pe = positional_encoding.compute_sinusoid_position_encoding(batch_size, max_length, dmodel)
    pe = pe[0]
    pe = pe.numpy().T
    print(pe.shape)
    plot_util.plotmesh(pe, title="sinusoid position encoding")
