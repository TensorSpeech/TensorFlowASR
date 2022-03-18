import tensorflow as tf

from tensorflow_asr.utils import math_util


def test():
    a = math_util.masked_fill(
        tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.float32),
        [[True, True, True], [True, False, True], [False, True, True]],
        value=-1e9,
    )
    print(a.numpy())
