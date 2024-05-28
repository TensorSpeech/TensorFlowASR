import os

from tensorflow_asr import tf
from tensorflow_asr.utils import file_util, math_util


def test_load_yaml():
    a = file_util.load_yaml(f"{os.path.dirname(__file__)}/../examples/conformer/config_wp.yml")
    print(a)


def test_mask_fill():
    a = math_util.masked_fill(
        tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32),
        [[True, True, True], [True, False, True], [False, True, True]],
        value=-1e9,
    )
    print(a.numpy())


def test_dataset():
    a = [1, 2, 3, 4, 5, 6, 7]
    batch = 2
    ds = tf.data.Dataset.from_tensor_slices(a)
    ds = ds.cache()
    ds = ds.shuffle(3)
    ds = ds.repeat(3)
    ds = ds.batch(batch, drop_remainder=True)
    print(list(ds.as_numpy_iterator()))
