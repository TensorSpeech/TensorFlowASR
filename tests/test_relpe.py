"""
Tests for relative positional encoding and the relative left shift.

`plot_util.plotmesh` used to end in `plt.show()`, which blocks on a GUI backend -- this file only
avoided hanging the suite because the shell running it had no window server. The figures are now
written into `$TFASR_PLOT_DIR` (`tests/figs/`, set by `tests/conftest.py`).
"""

import os

import numpy as np

from tensorflow_asr import tf
from tensorflow_asr.models.layers.multihead_attention import rel_left_shift
from tensorflow_asr.models.layers.positional_encoding import RelativeSinusoidalPositionalEncoding
from tensorflow_asr.utils import plot_util


def test_relative_sinusoidal_positional_encoding():
    batch_size, input_length, max_length, dmodel = 2, 300, 500, 144
    causal = False
    layer = RelativeSinusoidalPositionalEncoding(interleave=True, memory_length=input_length, causal=causal)
    _, pe = layer(
        (tf.random.normal([batch_size, max_length, dmodel]), tf.convert_to_tensor([input_length, input_length + 10])),
        training=False,
    )
    shift = tf.einsum("brd,btd->btr", pe, tf.ones([batch_size, max_length, dmodel]))
    shift = rel_left_shift(shift[0][None, None, ...], causal=causal)

    pe = tf.transpose(pe[0], perm=[1, 0]).numpy()
    shift = shift[0][0].numpy()

    assert pe.shape[0] == dmodel
    assert np.all(np.isfinite(pe))
    assert np.all(np.isfinite(shift))
    assert np.abs(pe).max() <= 1.0 + 1e-5, "a sinusoidal encoding is bounded by 1"

    for data, title, invert in ((pe, "sinusoid position encoding", False), (shift, "relshift", True)):
        path = plot_util.plotmesh(data, title=title, invert_yaxis=invert)
        assert os.path.exists(path), f"plotmesh did not write {path}"
        assert os.path.getsize(path) > 0


def test_plotmesh_default_path_uses_plot_dir(tmp_path, monkeypatch):
    """Without an explicit `output`, the figure lands in $TFASR_PLOT_DIR named after the title."""
    monkeypatch.setenv(plot_util.PLOT_DIR_ENV, str(tmp_path))
    path = plot_util.plotmesh(np.arange(12, dtype=np.float32).reshape(3, 4), title="my title")

    assert path == str(tmp_path / "my_title.png"), path
    assert os.path.getsize(path) > 0


def test_relshift():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[None, None, ...]
    shifted = tf.squeeze(tf.squeeze(rel_left_shift(a, causal=True), 0), 0).numpy()

    assert shifted.shape == (3, 3)
    assert np.array_equal(shifted, np.array([[3, 0, 4], [5, 6, 0], [7, 8, 9]]))

    # the non-causal variant keeps only the leading relative position
    non_causal = tf.squeeze(tf.squeeze(rel_left_shift(a, causal=False), 0), 0).numpy()
    assert non_causal.shape == (3, 1)
    assert np.array_equal(non_causal, np.array([[3], [5], [7]]))
