import tensorflow as tf

from tensorflow_asr.losses.rnnt_loss_naive import compute_rnnt_loss_and_grad_helper


def test():
    tf.config.run_functions_eagerly(True)
    logits = tf.random.normal([1, 20, 11, 10], dtype=tf.float32)
    labels = tf.range(10, dtype=tf.int32)[None, :]
    logit_length = tf.convert_to_tensor([20], dtype=tf.int32)
    label_length = tf.convert_to_tensor([10], dtype=tf.int32)

    compute_rnnt_loss_and_grad_helper(
        logits=logits,
        labels=labels,
        label_length=label_length,
        logit_length=logit_length,
        blank=0,
    )
