try:
    from keras.utils import tf_utils
except ImportError:
    from keras.src.utils import tf_utils


def convert_shapes(input_shape, to_tuples=True):
    if input_shape is None:
        return None
    return tf_utils.convert_shapes(input_shape, to_tuples=to_tuples)
