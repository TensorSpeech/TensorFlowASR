import collections

import numpy as np
import tensorflow as tf

from typing import Tuple
from tensorflow_asr.models.transducer.transducer_prediction import TransducerPrediction
from tensorflow_asr.models.transducer.transducer_joint import TransducerJoint

class BeamSearch:
    def __init__(self,
                 vocabulary_size: int,
                 predict_net: TransducerPrediction,
                 joint_net: TransducerJoint,
                 blank_token: int = 0,
                 beam_size: int = 1,
                 name="beam_search"):
        self._blank_token = blank_token
        self._vocabulary_size = vocabulary_size
        self._time_reduction_factor = 1
        self._predict_net = predict_net
        self._joint_net = joint_net
        self.beam_size = beam_size
        self._name = name
        self._predictor_states = None

    def call(self,
             encoded: tf.Tensor,
             encoded_length: tf.Tensor,
             parallel_iterations: int = 1,
             return_topk: bool = True):
        """
        RNN Transducer Beam Search
        Args:
            encoded (tf.Tensor): Tensor of encoder output with a shape [B, T, V]
            encoded_length (tf.Tensor): Tensor of input lengths (timewise) with shape [B]
            parallel_iterations (int): How many samples from the batch should be handled at once.
                                       Increasing it might result in higher memory usage.
            return_topk: If true, functions returns tensors of sizes [B, beam_size, T], [B, beam_size]
                         else it returns tensor of size [B, 1, T], [B, 1]
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: a batch of predictions tokens and a batch of probabilities
        """
        max_len = tf.reduce_max(encoded_length)
        batch_size = tf.shape(encoded)[0]
        start_batch = tf.constant(0)

        init_predictions_arr = tf.TensorArray(
            dtype=tf.int32,
            size=batch_size,
            dynamic_size=False,
            element_shape=[self.beam_size, None],
            clear_after_read=False,
        )

        init_probabilities_arr = tf.TensorArray(
            dtype=tf.float32,
            size=batch_size,
            dynamic_size=False,
            element_shape=[self.beam_size],
            clear_after_read=False,
        )

        def cond(current_batch, _prev_predictions, _prev_probabilities):
            return tf.less(current_batch, batch_size)

        def body(current_batch, prev_predictions, prev_probabilities):
            cur_probabilities, cur_predictions = self._perform_beam_search(
                encoded[current_batch],
                encoded_length[current_batch]
            )
            cur_predictions_padded = tf.pad(cur_predictions,
                                            [[0, 0], [0, max_len - tf.shape(cur_predictions)[1]]],
                                            constant_values=self._blank_token)
            predictions_arr = prev_predictions.write(current_batch, cur_predictions_padded)
            probabilities_arr = prev_probabilities.write(current_batch, cur_probabilities)

            return current_batch + 1, predictions_arr, probabilities_arr

        _batch, predictions_arr, probabilities_arr = tf.while_loop(
            cond,
            body,
            loop_vars=[start_batch, init_predictions_arr, init_probabilities_arr],
            parallel_iterations=parallel_iterations,
            swap_memory=True
        )
        predictions = predictions_arr.stack()
        probabilities = probabilities_arr.stack()

        if return_topk:
            return predictions, probabilities

        predictions = tf.slice(predictions, [0, 0, 0], [batch_size, 1, max_len])
        probabilities = tf.slice(probabilities, [0, 0], [batch_size, 1])

        return predictions, probabilities

    def _perform_beam_search(self, encoded: tf.Tensor, encoded_length: tf.Tensor) -> [tf.Tensor, tf.Tensor]:  # [T, V]
        encoded_expanded = tf.expand_dims(encoded, axis=1)  # [T, 1, V]
        encoded_beam_expanded = tf.tile(encoded_expanded, [1, self.beam_size, 1])  # [T, beam_size, V]

        max_time = encoded_length
        start_time = tf.constant(0)

        init_prediction_array = tf.TensorArray(
            dtype=tf.int32,
            size=max_time,
            dynamic_size=False,
            element_shape=tf.TensorShape(self.beam_size),
            clear_after_read=False,
        )
        init_encoder_prediction, init_path_probabilities, init_predictor_states = self._get_init_state()

        def cond(current_time, _path_probabilities, _last_predicted, _predictor_states, _last_encoder_prediction):
            return tf.less(current_time, max_time)

        def body(current_time, path_probabilities, prediction_array, predictor_states, last_encoder_prediction):
            new_path_probabilities, new_predictions, path_indices, new_predictor_states = self._joint_predictor_step(
                previous_path_probabilities=path_probabilities,
                predicted=last_encoder_prediction,
                encoded_slice=encoded_beam_expanded[current_time],
                states=predictor_states
            )
            blanks = tf.equal(new_predictions, self._blank_token)
            new_encoder_prediction = tf.where(blanks, last_encoder_prediction, new_predictions)
            prediction_tensor = prediction_array.stack()
            top_predictions_paths = tf.gather(prediction_tensor, tf.reshape(path_indices, [-1]), axis=1)
            prediction_array = prediction_array.unstack(top_predictions_paths)
            prediction_array = prediction_array.write(current_time, new_predictions)

            return current_time + 1, new_path_probabilities, prediction_array, new_predictor_states, new_encoder_prediction

        _time, probabilities, predictions, _states, _last_prediction = tf.while_loop(
            cond,
            body,
            loop_vars=[start_time,
                       init_path_probabilities,
                       init_prediction_array,
                       init_predictor_states,
                       init_encoder_prediction],
            parallel_iterations=1,
            swap_memory=True
        )

        return tf.transpose(probabilities), tf.transpose(predictions.stack())

    def _get_init_state(self) -> Tuple[tf.TensorArray, tf.Tensor, tf.Tensor]:
        # First iteration of beam search has to start from a single node, so
        # all of the beam search paths don't infer the same path
        init_path_probabilities = tf.ones(self.beam_size - 1, dtype=tf.float32) * -np.inf
        init_path_probabilities = tf.pad(init_path_probabilities, [[1, 0]])
        init_predictor_states = self._predict_net.get_initial_state(
            batch_size=tf.constant(self.beam_size))  # [num_rnns, 2, beam_size, P]
        init_predictions = tf.ones(self.beam_size, dtype=tf.int32) * self._blank_token  # [beam_size]

        return init_predictions, init_path_probabilities, init_predictor_states

    def _joint_predictor_step(self,
                              previous_path_probabilities: tf.Tensor,  # [beam_size]
                              encoded_slice: tf.Tensor,  # [beam_size, V]
                              predicted: tf.Tensor,  # [beam_size]
                              states: tf.Tensor  # [num_rnns, 2, beam_size, V]
                              ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        encoded = tf.expand_dims(encoded_slice, axis=1)  # [beam_size, 1, V]
        last_predicted = tf.expand_dims(predicted, axis=1)
        prediction, new_states = self._predict_net.recognize(last_predicted, states)
        joint_out = tf.squeeze(self._joint_net([encoded, prediction], training=False), [1, 2])
        logits = tf.nn.log_softmax(joint_out)  # [beam_size, alphabet_size]

        new_path_probabilities, indices = self._get_topk_paths(logits, previous_path_probabilities)
        new_states_masked = self._get_new_states_masked(states, new_states, indices)
        path_indices, new_predictions = tf.split(indices, 2, axis=1)
        new_predictions_unpacked = tf.squeeze(new_predictions, axis=1)

        return new_path_probabilities, new_predictions_unpacked, path_indices, new_states_masked

    def _get_topk_paths(self,
                        logits: tf.Tensor,  # [beam_size, alphabet_size]
                        path_probabilities: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:  # [beam_size]
        current_path_probabilities = tf.expand_dims(path_probabilities, axis=1)  # [beam_size, 1]
        current_path_probabilities = tf.tile(current_path_probabilities,
                                             [1, tf.shape(logits)[1]])  # [beam_size, alphabet_size]
        current_path_probabilities = logits + current_path_probabilities
        flat_current_path_probabilities = tf.reshape(current_path_probabilities, [-1])
        topk_paths = tf.math.top_k(flat_current_path_probabilities, k=self.beam_size)
        indices = tf.transpose(tf.unravel_index(topk_paths.indices, dims=tf.shape(logits)))
        new_path_probabilities = tf.gather_nd(current_path_probabilities, indices) + tf.gather_nd(logits, indices)

        return new_path_probabilities, indices

    def _get_new_states_masked(self,
                               old_states: tf.Tensor,  # [num_rnns, 2, beam_size, V]
                               new_states: tf.Tensor,  # [num_rnns, 2, beam_size, V]
                               topk_indices: tf.Tensor) -> tf.Tensor:  # [beam_size, alphabet_size]
        new_path_indices, new_predictions = tf.split(topk_indices, 2, axis=1)
        change_state_mask = tf.not_equal(new_predictions, self._blank_token)

        old_states_batch_major = tf.transpose(old_states, perm=[2, 0, 1, 3])  # [beam_size, num_rnns, 2, V]
        new_states_batch_major = tf.transpose(new_states, perm=[2, 0, 1, 3])  # [beam_size, num_rnns, 2, V]
        old_paths_states = tf.gather_nd(old_states_batch_major, new_path_indices)
        old_paths_states = tf.transpose(old_paths_states, perm=[1, 2, 0, 3])  # [num_rnns, 2, beam_size, V]
        new_paths_states = tf.gather_nd(new_states_batch_major, new_path_indices)
        new_paths_states = tf.transpose(new_paths_states, perm=[1, 2, 0, 3])  # [num_rnns, 2, beam_size, V]

        new_states_masked = tf.where(change_state_mask, new_paths_states, old_paths_states)

        return new_states_masked
