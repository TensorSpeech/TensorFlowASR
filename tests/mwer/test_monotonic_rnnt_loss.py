import unittest
from typing import Optional, Callable
import tensorflow as tf
import numpy as np
import os
from tensorflow_asr.mwer.monotonic_rnnt_loss import monotonic_rnnt_loss, MonotonicRnntData

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def finite_difference_gradient(func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Approximates gradient numerically

    Source: https://github.com/alexeytochin/tf_seq2seq_losses/
    """
    input_shape = tf.shape(x)[1:]
    input_rank = input_shape.shape[0]
    dim = tf.reduce_prod(input_shape)
    dx = tf.reshape(tf.eye(dim, dtype=x.dtype), shape=tf.concat([tf.constant([1]), tf.reshape(dim, [1]), input_shape], axis=0))
    # shape = [1, dim] + input_shape

    pre_x1 = tf.expand_dims(x, 1) + epsilon * dx
    # shape = [batch_size, dim] + input_shape
    x1 = tf.reshape(pre_x1, shape=tf.concat([tf.constant([-1], dtype=tf.int32), input_shape], axis=0))
    # shape = [batch_size * dim] + input_shape
    x0 = tf.tile(x, multiples=[dim] + [1] * input_rank)

    pre_derivative = (func(x1) - func(x0)) / epsilon
    # shape = [batch_size * dim]
    derivative = tf.reshape(pre_derivative, shape=tf.concat([tf.constant([-1]), input_shape], axis=0))
    # shape = [batch_size] + input_shape
    return derivative


def generate_inputs():
    labels_len = tf.convert_to_tensor([6])
    inputs_len = tf.convert_to_tensor([10])
    vocab_size = 4
    batch_size = 1
    labels = tf.convert_to_tensor([[2, 1, 3, 2, 3, 2, 0, 0]])

    max_u = 8
    max_t = 12

    logits = tf.random.uniform(shape=[batch_size, max_t, max_u + 1, vocab_size], minval=0.1, maxval=0.8, seed=42)
    return logits, labels, labels_len, inputs_len


class TestRnntLoss(unittest.TestCase):
    def assert_tensors_almost_equal(self, first: tf.Tensor, second: tf.Tensor, places: Optional[int]):
        self.assertAlmostEqual(first=0, second=tf.norm(first - second, ord=np.inf).numpy(), places=places)

    def test_alpha_beta(self):
        """Checks whether bottom left element of beta == top right element of alpha * prob_blank

        From the definition of forward bacward variables the loss should be equal to
        the bottom left element of beta and top right element of alpha multiplied by the probability to output blank.
        """
        logits, labels, labels_len, inputs_len = generate_inputs()
        loss_data = MonotonicRnntData(logits, labels, inputs_len, labels_len)

        beta_final = loss_data.log_loss
        alpha_final = (
            loss_data.alpha[:, inputs_len[0] - 1, labels_len[0]] + loss_data.blank_probs[:, inputs_len[0] - 1, labels_len[0]]
        )

        self.assert_tensors_almost_equal(beta_final, alpha_final, 3)

    def test_gradient_with_finite_difference(self):
        logits, labels, labels_len, inputs_len = generate_inputs()

        def loss_fn(logit):
            return tf.reduce_sum(monotonic_rnnt_loss(tf.expand_dims(logit, 0), labels, labels_len, inputs_len))

        gradient_numerical = finite_difference_gradient(
            func=lambda logits_: tf.vectorized_map(fn=loss_fn, elems=logits_), x=logits, epsilon=1e-2
        )

        with tf.GradientTape() as tape:
            tape.watch([logits])
            loss = loss_fn(logits[0])

        gradient_analytic = tape.gradient(loss, sources=logits)

        self.assert_tensors_almost_equal(gradient_numerical, gradient_analytic, 1)

    def test_small_gradient(self):
        logits = tf.constant(
            [
                [
                    [[0.1, 0.6, 0.1], [0.8, 0.2, 0.3], [0.7, 0.2, 0.1]],
                    [[0.6, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.8, 0.3]],
                    [[0.1, 0.7, 0.1], [0.1, 0.2, 0.6], [0.1, 0.9, 0.3]],
                ]
            ],
            dtype=tf.float32,
        )

        labels = tf.convert_to_tensor([[1, 2]])
        labels_len = tf.convert_to_tensor([2])
        inputs_len = tf.convert_to_tensor([3])

        expected_loss = tf.constant([-3.5545435])

        loss_data = MonotonicRnntData(logits, labels, inputs_len, labels_len)

        self.assert_tensors_almost_equal(loss_data.log_loss, expected_loss, places=6)
