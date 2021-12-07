import collections
from typing import Dict

import tensorflow as tf

from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.models.transducer.base_transducer import TransducerPrediction, TransducerJoint
from tensorflow_asr.utils import math_util, shape_util

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))
BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))


class BeamSearch:
    def __init__(self,
                 text_featurizer: TextFeaturizer,
                 encoder: tf.keras.Model,
                 predict_net: TransducerPrediction,
                 joint_net: TransducerJoint,
                 name="transducer"):
        self._time_reduction_factor = 1
        self._text_featurizer = text_featurizer
        self._encoder = encoder
        self._predict_net = predict_net
        self._joint_net = joint_net
        self._name = name

    @tf.function
    def recognize_beam(
        self,
        inputs: Dict[str, tf.Tensor],
        lm: bool = False,
    ):
        """
        RNN Transducer Beam Search
        Args:
            inputs (Dict[str, tf.Tensor]): Input dictionary containing "inputs" and "inputs_length"
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self._encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self._time_reduction_factor)
        return self._perform_beam_search_batch(
            encoded=encoded,
            encoded_length=encoded_length,
            lm=lm,
        )

    def _decoder_inference(
        self,
        encoded: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        tflite: bool = False,
    ):
        """Infer function for decoder

        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self._name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self._predict_net.recognize(predicted, states, tflite=tflite)  # [1, 1, P], states
            ytu = tf.nn.log_softmax(self._joint_net([encoded, y], training=False))  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    def _perform_beam_search_batch(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        lm: bool = False,
        parallel_iterations: int = 10,
        swap_memory: bool = True,
    ):
        with tf.name_scope(f"{self._name}_perform_beam_search_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=None,
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_beam_search(
                    encoded[batch],
                    encoded_length[batch],
                    lm,
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self._text_featurizer.blank)
            return self._text_featurizer.iextract(decoded.stack())

    def _perform_beam_search(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        lm: bool = False,
        parallel_iterations: int = 10,
        swap_memory: bool = True,
        tflite: bool = False,
    ):
        with tf.name_scope(f"{self._name}_beam_search"):
            beam_width = tf.cond(
                tf.less(self._text_featurizer.decoder_config.beam_width, self._text_featurizer.num_classes),
                true_fn=lambda: self._text_featurizer.decoder_config.beam_width,
                false_fn=lambda: self._text_featurizer.num_classes - 1,
            )
            total = encoded_length

            def initialize_beam(dynamic=False):
                return BeamHypothesis(
                    score=tf.TensorArray(
                        dtype=tf.float32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]),
                        clear_after_read=False,
                    ),
                    indices=tf.TensorArray(
                        dtype=tf.int32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]),
                        clear_after_read=False,
                    ),
                    prediction=tf.TensorArray(
                        dtype=tf.int32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=None,
                        clear_after_read=False,
                    ),
                    states=tf.TensorArray(
                        dtype=tf.float32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape(shape_util.shape_list(self._predict_net.get_initial_state())),
                        clear_after_read=False,
                    ),
                )

            B = initialize_beam()
            B = BeamHypothesis(
                score=B.score.write(0, 0.0),
                indices=B.indices.write(0, self._text_featurizer.blank),
                prediction=B.prediction.write(0, tf.ones([total], dtype=tf.int32) * self._text_featurizer.blank),
                states=B.states.write(0, self._predict_net.get_initial_state()),
            )

            def condition(time, total, B):
                return tf.less(time, total)

            def body(time, total, B):
                A = initialize_beam(dynamic=True)
                A = BeamHypothesis(
                    score=A.score.unstack(B.score.stack()),
                    indices=A.indices.unstack(B.indices.stack()),
                    prediction=A.prediction.unstack(
                        math_util.pad_prediction_tfarray(B.prediction, blank=self._text_featurizer.blank).stack()
                    ),
                    states=A.states.unstack(B.states.stack()),
                )
                A_index = tf.constant(0, tf.int32)
                B = initialize_beam()

                encoded_t = tf.gather_nd(encoded, tf.expand_dims(time, axis=-1))

                def beam_condition(beam, beam_width, _A, _A_index, _B):
                    return tf.less(beam, beam_width)

                def beam_body(beam, beam_width, A, A_index, B):
                    # get y_hat
                    y_hat_score, y_hat_score_index = tf.math.top_k(A.score.stack(), k=1, sorted=True)
                    y_hat_score = y_hat_score[0]
                    y_hat_index = tf.gather_nd(A.indices.stack(), y_hat_score_index)
                    y_hat_prediction = tf.gather_nd(
                        math_util.pad_prediction_tfarray(A.prediction, blank=self._text_featurizer.blank).stack(),
                        y_hat_score_index,
                    )
                    y_hat_states = tf.gather_nd(A.states.stack(), y_hat_score_index)

                    # remove y_hat from A
                    remain_indices = tf.range(0, tf.shape(A.score.stack())[0], dtype=tf.int32)
                    remain_indices = tf.gather_nd(remain_indices, tf.where(tf.not_equal(remain_indices, y_hat_score_index[0])))
                    remain_indices = tf.expand_dims(remain_indices, axis=-1)
                    A = BeamHypothesis(
                        score=A.score.unstack(tf.gather_nd(A.score.stack(), remain_indices)),
                        indices=A.indices.unstack(tf.gather_nd(A.indices.stack(), remain_indices)),
                        prediction=A.prediction.unstack(
                            tf.gather_nd(
                                math_util.pad_prediction_tfarray(A.prediction, blank=self._text_featurizer.blank).stack(),
                                remain_indices,
                            )
                        ),
                        states=A.states.unstack(tf.gather_nd(A.states.stack(), remain_indices)),
                    )
                    A_index = tf.cond(tf.equal(A_index, 0), true_fn=lambda: A_index, false_fn=lambda: A_index - 1)

                    ytu, new_states = self._decoder_inference(
                        encoded=encoded_t, predicted=y_hat_index, states=y_hat_states, tflite=tflite
                    )

                    def predict_condition(pred, _A, _A_index, _B):
                        return tf.less(pred, self._text_featurizer.num_classes)

                    def predict_body(pred, A, A_index, B):
                        new_score = y_hat_score + tf.gather_nd(ytu, tf.expand_dims(pred, axis=-1))

                        def true_fn():
                            return (
                                B.score.write(beam, new_score),
                                B.indices.write(beam, y_hat_index),
                                B.prediction.write(beam, y_hat_prediction),
                                B.states.write(beam, y_hat_states),
                                A.score,
                                A.indices,
                                A.prediction,
                                A.states,
                                A_index,
                            )

                        def false_fn():
                            scatter_index = math_util.count_non_blank(y_hat_prediction, blank=self._text_featurizer.blank)
                            updated_prediction = tf.tensor_scatter_nd_update(
                                y_hat_prediction,
                                indices=tf.reshape(scatter_index, [1, 1]),
                                updates=tf.expand_dims(pred, axis=-1),
                            )
                            return (
                                B.score,
                                B.indices,
                                B.prediction,
                                B.states,
                                A.score.write(A_index, new_score),
                                A.indices.write(A_index, pred),
                                A.prediction.write(A_index, updated_prediction),
                                A.states.write(A_index, new_states),
                                A_index + 1,
                            )

                        b_score, b_indices, b_prediction, b_states, a_score, a_indices, a_prediction, a_states, A_index = tf.cond(
                            tf.equal(pred, self._text_featurizer.blank), true_fn=true_fn, false_fn=false_fn
                        )

                        B = BeamHypothesis(score=b_score, indices=b_indices, prediction=b_prediction, states=b_states)
                        A = BeamHypothesis(score=a_score, indices=a_indices, prediction=a_prediction, states=a_states)

                        return pred + 1, A, A_index, B

                    _, A, A_index, B = tf.while_loop(
                        predict_condition,
                        predict_body,
                        loop_vars=[0, A, A_index, B],
                        parallel_iterations=parallel_iterations,
                        swap_memory=swap_memory,
                    )

                    return beam + 1, beam_width, A, A_index, B

                _, _, A, A_index, B = tf.while_loop(
                    beam_condition,
                    beam_body,
                    loop_vars=[0, beam_width, A, A_index, B],
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )

                return time + 1, total, B

            _, _, B = tf.while_loop(
                condition,
                body,
                loop_vars=[0, total, B],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            scores = B.score.stack()
            prediction = math_util.pad_prediction_tfarray(B.prediction, blank=self._text_featurizer.blank).stack()
            if self._text_featurizer.decoder_config.norm_score:
                prediction_lengths = math_util.count_non_blank(prediction, blank=self._text_featurizer.blank, axis=1)
                scores /= tf.cast(prediction_lengths, dtype=scores.dtype)

            y_hat_score, y_hat_score_index = tf.math.top_k(scores, k=1)
            y_hat_score = y_hat_score[0]
            y_hat_index = tf.gather_nd(B.indices.stack(), y_hat_score_index)
            y_hat_prediction = tf.gather_nd(prediction, y_hat_score_index)
            y_hat_states = tf.gather_nd(B.states.stack(), y_hat_score_index)

            return Hypothesis(index=y_hat_index, prediction=y_hat_prediction, states=y_hat_states)
