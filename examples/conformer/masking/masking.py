import tensorflow as tf
from tensorflow_asr.utils.utils import shape_list


def create_padding_mask(features, input_length, time_reduction_factor):
    """
    Create masking with 0 for paddings and 1 for non-paddings
    Args:
        features ([tf.Tensor]): audio features with shape [B, T, F, C]
        input_length ([tf.Tensor]): audio features length with shape [B]
        time_reduction_factor ([int])

    Returns:
        [tf.Tensor]: with shape [B, Tquery, Tkey]
    """
    batch_size, padded_time, _, _ = shape_list(features)
    reduced_padded_time = tf.math.ceil(padded_time / time_reduction_factor)

    def create_mask(length):
        reduced_length = tf.math.ceil(length / time_reduction_factor)
        mask = tf.ones([reduced_length, reduced_length], dtype=tf.float32)
        return tf.pad(
            mask,
            [
                [0, reduced_padded_time - reduced_length],
                [0, reduced_padded_time - reduced_length]
            ],
            mode="CONSTANT",
            constant_values=0.0
        )

    return tf.map_fn(create_mask, input_length, fn_output_signature=tf.TensorSpec([None, None], dtype=tf.float32))
