from typing import List, Tuple

import tensorflow as tf
import unittest
import os

from tensorflow_asr.mwer.beam_search import BeamSearch

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class BeamSearchTest(tf.test.TestCase):
    def test_beam_width_3(self):  # batch_size = 1
        class PredictNet:
            def __init__(self,
                         alphabet_size: int,
                         favoured_paths: List[tf.Tensor],
                         max_path_size=1):
                no_nodes = tf.math.pow(alphabet_size, max_path_size)
                self._edge_map = tf.zeros([no_nodes, alphabet_size])
                self._alphabet_size = alphabet_size
                favoured_paths = favoured_paths - tf.constant(1)

                for path in favoured_paths:
                    token_list = tf.unstack(path)
                    current_node = tf.constant(0)
                    for token in token_list:
                        if tf.less(token, 0):
                            continue
                        self._edge_map = tf.tensor_scatter_nd_update(self._edge_map,
                                                                     [[current_node, token]],
                                                                     [1.0])
                        current_node = self._get_next_node_index(token, current_node)

            def _get_next_node_index(self, token, current_node):
                if tf.less(token, 0):
                    return tf.zeros([], dtype=tf.int32)
                return tf.squeeze(token + current_node * self._alphabet_size + 1)

            def get_initial_state(self, batch_size: tf.Tensor):
                return tf.reshape(tf.zeros(batch_size, dtype=tf.int32), [1, 1, -1, 1])

            def recognize(self, last_predicted: tf.Tensor, states: tf.Tensor):
                last_predicted = tf.unstack(last_predicted - tf.constant(1))
                states = tf.unstack(tf.squeeze(states))
                new_states = []

                for state, prediction in zip(states, last_predicted):
                    new_states.append(self._get_next_node_index(prediction, state))

                new_states = tf.stack(new_states)
                new_predictions = tf.gather(self._edge_map, new_states)

                return new_predictions, tf.reshape(new_states, [1, 1, -1, 1])

        class JointNet:
            def __call__(self, input: Tuple[tf.Tensor, tf.Tensor], training: bool):
                encoded, prediction = input
                encoded = tf.expand_dims(encoded, axis=1)
                prediction = tf.pad(prediction, [[0, 0], [1, 0]])
                prediction = tf.expand_dims(prediction, axis=1)
                prediction = tf.expand_dims(prediction, axis=1)

                return encoded + prediction

        expected_predictions = tf.constant(
            [[
                [1, 2, 0],
                [1, 2, 2],
                [1, 2, 3]
            ]],
            dtype=tf.int32
        )

        vocab_size = 4
        blank_token = 0
        beam_size = 3
        favoured_paths = tf.constant(
            [
                [1, 2, 0],
                [1, 2, 2],
                [1, 2, 3]
            ]
        )
        encoded = tf.constant([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1.01, 0, 0, 0],
        ])
        encoded = tf.expand_dims(encoded, axis=0)
        encoded_length = tf.constant([3])

        predict_net = PredictNet(vocab_size - 1, favoured_paths, max_path_size=3)
        joint_net = JointNet()

        beam = BeamSearch(vocabulary_size=vocab_size,
                          predict_net=predict_net,
                          joint_net=joint_net,
                          blank_token=blank_token,
                          beam_size=beam_size)
        predictions, _probabilities = beam.call(encoded, encoded_length)

        self.assertAllEqual(predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()
