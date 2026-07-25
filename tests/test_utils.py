import os

import numpy as np

from tensorflow_asr import tf
from tensorflow_asr.utils import file_util, math_util

REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_load_yaml():
    """`examples/conformer/config_wp.yml` is long gone, and load_yaml now needs `repodir`."""
    config = file_util.load_yaml(
        os.path.join(REPO_ROOT, "examples", "datasets", "librispeech", "characters", "char.yml.j2"),
        repodir=REPO_ROOT,
    )
    assert "decoder_config" in config
    assert config["decoder_config"]["type"] == "characters"


def test_load_yaml_renders_repodir():
    """The `{{repodir}}` placeholder must be substituted, not left as literal text."""
    config = file_util.load_yaml(
        os.path.join(REPO_ROOT, "examples", "datasets", "librispeech", "sentencepiece", "sp.yml.j2"),
        repodir=REPO_ROOT,
    )
    vocabulary = config["decoder_config"]["vocabulary"]
    assert "{{" not in vocabulary and "repodir" not in vocabulary
    assert vocabulary.startswith(REPO_ROOT)
    assert os.path.exists(vocabulary), f"config points at a missing vocabulary: {vocabulary}"


def test_mask_fill():
    filled = math_util.masked_fill(
        tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32),
        [[True, True, True], [True, False, True], [False, True, True]],
        value=-1e9,
    ).numpy()

    # True keeps the original value, False is replaced
    assert np.array_equal(filled[0], [1.0, 2.0, 3.0])
    assert filled[1][1] == -1e9 and np.array_equal(filled[1][[0, 2]], [4.0, 6.0])
    assert filled[2][0] == -1e9 and np.array_equal(filled[2][[1, 2]], [8.0, 9.0])


def test_dataset():
    values = [1, 2, 3, 4, 5, 6, 7]
    batch_size, repeats = 2, 3
    dataset = tf.data.Dataset.from_tensor_slices(values).cache().shuffle(3).repeat(repeats)
    batches = list(dataset.batch(batch_size, drop_remainder=True).as_numpy_iterator())

    assert len(batches) == len(values) * repeats // batch_size
    assert all(batch.shape == (batch_size,) for batch in batches)
    # drop_remainder discards at most one element per repeat cycle, never a whole value's worth
    flat = np.concatenate(batches)
    assert set(flat.tolist()) <= set(values)


def test_split_batch():
    """[B, ...] is regrouped into [ga_steps, mini_batch_size, ...] without reordering."""
    mini_batch_size, ga_steps = 4, 3
    tensor = tf.reshape(tf.range(mini_batch_size * ga_steps * 2 * 4, dtype=tf.float32), (12, 2, 4))
    grouped = math_util.split_tensor_by_ga(tensor, mini_batch_size, ga_steps)

    assert tuple(grouped.shape) == (ga_steps, mini_batch_size, 2, 4)
    # regrouping is a pure reshape: flattening the first two axes recovers the input exactly
    assert np.array_equal(tf.reshape(grouped, tf.shape(tensor)).numpy(), tensor.numpy())
    assert np.array_equal(grouped[0].numpy(), tensor[:mini_batch_size].numpy())
    assert np.array_equal(grouped[-1].numpy(), tensor[-mini_batch_size:].numpy())
