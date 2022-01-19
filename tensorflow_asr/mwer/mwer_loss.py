import tensorflow as tf
from typing import Callable, Dict

from tensorflow_asr.mwer.monotonic_rnnt_loss import monotonic_rnnt_loss
from tensorflow_asr.losses.rnnt_loss import rnnt_loss


class MWERLoss:
    def __init__(self,
                 risk_obj: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 global_batch_size=None,
                 blank=0,
                 name=None):
        self._risk_obj = risk_obj
        self._global_batch_size = global_batch_size
        self._blank = blank
        self._name = name

    def __call__(self,
                 prediction: Dict[str, tf.Tensor],
                 label: Dict[str, tf.Tensor],
                 hypotheses: Dict[str, tf.Tensor],
                 ) -> tf.Tensor:
        risk_vals = tf.py_function(self._risk_obj,
                                   [hypotheses["sentences"], hypotheses["labels"]],
                                   Tout=tf.float32)

        loss = mwer_loss(
            prediction["logits"],
            risk_vals,
            hypotheses["log_probas"],
            label["labels"],
            prediction["logits_length"],
            label["labels_length"],
            self._blank,
            # self._name,
        )

        return tf.nn.compute_average_loss(loss, global_batch_size=self._global_batch_size)


@tf.function(experimental_relax_shapes=True)
def mwer_loss(
        logits: tf.Tensor,  # [batch_size * beam_size, T, U, V]
        risk_vals: tf.Tensor,  # [batch_size * beam_size]
        hypotheses_log_probas: tf.Tensor,  # [batch_size * beam_size]
        labels: tf.Tensor,  # [batch_size * beam_size]
        logit_length: tf.Tensor,  # [batch_size * beam_size]
        label_length: tf.Tensor,  # [batch_size * beam_size]
        blank: int = 0,
        # name="mwer_loss"
):
    @tf.custom_gradient
    def compute_grads(input: tf.Tensor):
        softmax_probas = tf.nn.softmax(hypotheses_log_probas)
        expected_risk = tf.reduce_sum(softmax_probas * risk_vals)

        probas_normalized = tf.nn.softmax(hypotheses_log_probas)
        risk_diffs = risk_vals - expected_risk
        lhs = probas_normalized * risk_diffs

        with tf.GradientTape() as tape:
            tape.watch(input)
            rnn_loss_val = monotonic_rnnt_loss(input,
                                               labels,
                                               label_length,
                                               logit_length,
                                               blank)
        rhs = tape.gradient(rnn_loss_val, input)
        grad_val = tf.reshape(lhs, [-1, 1, 1, 1]) * rhs

        def grad(init_grad):
            grads = [tf.reshape(init_grad, shape=[-1, 1, 1, 1]) * grad_val]
            return grads

        return expected_risk, grad

    return compute_grads(logits)
