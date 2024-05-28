from tensorflow_asr import tf
from tensorflow_asr.models.layers.multihead_attention import rel_left_shift
from tensorflow_asr.models.layers.positional_encoding import RelativeSinusoidalPositionalEncoding
from tensorflow_asr.utils import plot_util


def test():
    batch_size, input_length, max_length, dmodel = 2, 300, 500, 144
    causal = False
    layer = RelativeSinusoidalPositionalEncoding(interleave=True, memory_length=input_length, causal=causal)
    _, pe = layer((tf.random.normal([batch_size, max_length, dmodel]), tf.convert_to_tensor([input_length, input_length + 10])), training=False)
    shift = tf.einsum("brd,btd->btr", pe, tf.ones([batch_size, max_length, dmodel]))
    shift = rel_left_shift(shift[0][None, None, ...], causal=causal)
    pe = tf.transpose(pe[0], perm=[1, 0])
    pe = pe.numpy()
    print(pe.shape)
    shift = shift[0][0]
    shift = shift.numpy()
    print(shift.shape)
    plot_util.plotmesh(pe, title="sinusoid position encoding", invert_yaxis=False)
    plot_util.plotmesh(shift, title="relshift")


def test_relshift():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)
    a = a[None, ...]
    a = a[None, ...]
    b = rel_left_shift(a, causal=True)
    b = tf.squeeze(b, 0)
    b = tf.squeeze(b, 0)
    print(b)
