import tensorflow as tf
from typing import List, Callable

from tensorflow_asr.mwer.beam_search import BeamHypothesis


class MWERLoss:
    def __init__(self, risk_obj: Callable[[List[tf.Tensor], tf.Tensor], tf.Tensor]):
        self._risk_obj = risk_obj

    def __call__(self, hypotheses: List[BeamHypothesis], true_label: tf.Tensor) -> tf.Tensor:
        hypotheses_labels = [h.indices.stack() for h in hypotheses]
        probas_list = [h.score for h in hypotheses]

        risk_vals = self._risk_obj(hypotheses_labels, true_label)
        probas = tf.convert_to_tensor(probas_list, dtype=tf.float32)

        return self._mwer_lhs(probas=probas, risk_vals=risk_vals)


    @tf.function
    def _mwer_lhs(self, probas: tf.Tensor, risk_vals: tf.Tensor) -> tf.Tensor:
        log_probas = tf.math.log(probas)
        probas_normalized = tf.nn.softmax(log_probas)

        mean_risk = tf.reduce_mean(risk_vals)
        risk_diffs = tf.cast((risk_vals - mean_risk), dtype=tf.float32)

        grads = probas_normalized * risk_diffs

        return grads
