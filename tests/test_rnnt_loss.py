import tensorflow as tf

from tensorflow_asr.losses.rnnt_loss_naive import compute_rnnt_loss_and_grad_helper

B = 1
T = 30
U = 30
V = 1000
blank = 0


@tf.function
def run():
    logits = tf.random.normal([B, T, U + 1, V], dtype=tf.float32)
    labels = tf.repeat(tf.range(U, dtype=tf.int32)[None, :], B, 0)
    logit_length = tf.repeat(tf.convert_to_tensor([T], dtype=tf.int32), B, 0)
    label_length = tf.repeat(tf.convert_to_tensor([U], dtype=tf.int32), B, 0)

    loss, grad = compute_rnnt_loss_and_grad_helper(
        logits=logits,
        labels=labels,
        label_length=label_length,
        logit_length=logit_length,
        blank=blank,
    )
    tf.print(loss, grad)


def test():
    tf.config.run_functions_eagerly(False)
    run()
