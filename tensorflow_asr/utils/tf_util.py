import importlib

import tensorflow as tf

from tensorflow_asr.utils.env_util import KERAS_SRC

tf_utils = importlib.import_module(f"{KERAS_SRC}.utils.tf_utils")


def convert_shapes(input_shape, to_tuples=True):
    if input_shape is None:
        return None

    def _is_shape_component(value):
        return value is None or isinstance(value, (int, tf.compat.v1.Dimension))

    def _is_atomic_shape(input_shape):
        # Ex: TensorShape or (None, 10, 32) or 5 or `None`
        if _is_shape_component(input_shape):
            return True
        if isinstance(input_shape, tf.TensorShape):
            return True
        if isinstance(input_shape, (tuple, list)) and all(_is_shape_component(ele) for ele in input_shape):
            return True
        return False

    def _convert_shape(input_shape):
        if input_shape is None:
            return None
        input_shape = tf.TensorShape(input_shape)
        if to_tuples:
            input_shape = tuple(input_shape.as_list())
        return input_shape

    return tf_utils.map_structure_with_atomic(_is_atomic_shape, _convert_shape, input_shape)
