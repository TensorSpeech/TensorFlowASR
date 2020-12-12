import tensorflow as tf
from tensorflow_asr.utils.utils import shape_list


def create_padding_mask(features, input_length):
    """
    Create masking with 0 for paddings and 1 for non-paddings
    Args:
        features ([tf.Tensor]): audio features with shape [B, T, F, C]
        input_length ([tf.Tensor]): audio features length with shape [B]
        dmodel ([int]): model size for attention

    Returns:
        [tf.Tensor]: with shape [B, Tquery, Tkey]
    """
    batch_size, padded_time, _, _ = shape_list(features)

    def create_mask(length):
        mask = tf.ones([length, length], dtype=tf.float32)
        return tf.pad(mask, [[0, padded_time - length], [0, padded_time - length]], mode="CONSTANT", constant_values=0.0)

    return tf.map_fn(create_mask, input_length, fn_output_signature=tf.TensorSpec([None, None], dtype=tf.float32))
