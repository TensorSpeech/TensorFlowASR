import tensorflow as tf

from tensorflow_asr.losses.rnnt_loss_naive import compute_rnnt_loss_and_grad_helper

B = 1
T = 20
U = 10
V = 1000
blank = 0


@tf.function
def run():
    logits = tf.random.normal([B, T, U + 1, V], dtype=tf.float32)
    labels = tf.range(U, dtype=tf.int32)[None, :]
    logit_length = tf.convert_to_tensor([T], dtype=tf.int32)
    label_length = tf.convert_to_tensor([U], dtype=tf.int32)

    compute_rnnt_loss_and_grad_helper(
        logits=logits,
        labels=labels,
        label_length=label_length,
        logit_length=logit_length,
        blank=blank,
    )


def test():
    tf.config.run_functions_eagerly(False)
    run()
