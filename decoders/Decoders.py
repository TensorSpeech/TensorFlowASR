from __future__ import absolute_import

import tensorflow as tf
import numpy as np


class Decoder:
    def __init__(self, index_to_token, blank_index=-1):
        self.index_to_token = index_to_token
        # Blank index default -1: https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode?hl=en
        self.blank_index = blank_index

    def convert_to_string(self, decoded):
        # Remove blank indices
        def map_cvrt(elem):
            elem = np.array(elem)
            elem = elem[elem != self.blank_index]
            return ''.join([self.index_to_token[i] for i in elem])

        # Convert to string
        decoded = tf.map_fn(map_cvrt, decoded)
        return decoded.numpy()

    def decode(self, probs, input_length):
        pass


class GreedyDecoder(Decoder):
    """ Decode the best guess from probs using greedy algorithm """

    def decode(self, probs, input_length):
        # probs.shape = [batch_size, time_steps, num_classes]
        decoded = tf.keras.backend.ctc_decode(y_pred=probs, input_length=input_length, greedy=True)
        # decoded shape = [batch_size, decoded index]
        decoded = tf.convert_to_tensor(decoded)
        return self.convert_to_string(decoded)


class BeamSearchDecoder(Decoder):
    """ Decode probs using beam search algorithm """

    def __init__(self, index_to_token, beam_width=1024, lm_path=None, blank_index=-1):
        super().__init__(index_to_token, blank_index)
        self.beam_width = beam_width
        self.lm_path = lm_path

    def decode(self, probs, input_length):
        # probs.shape = [batch_size, time_steps, num_classes]
        decoded = tf.keras.backend.ctc_decode(y_pred=probs, input_length=input_length, greedy=False,
                                              beam_width=self.beam_width)
        # decoded shape = [batch_size, top_path=1, decoded index]
        decoded = tf.squeeze(decoded)
        return self.convert_to_string(decoded)
