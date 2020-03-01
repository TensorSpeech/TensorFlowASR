from __future__ import absolute_import

import tensorflow as tf
import numpy as np


class Decoder:
    def __init__(self, index_to_token):
        self.index_to_token = index_to_token
        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
        # default blank index changed from 0 to -1
        self.blank_index = -1

    def convert_to_string(self, decoded):
        # Remove blank indices
        def map_cvrt(elem):
            elem = np.array(elem)
            elem = elem[elem != self.blank_index]
            return ''.join([self.index_to_token[i] for i in elem])

        # Convert to string
        return tf.map_fn(map_cvrt, decoded, dtype=tf.string)

    def decode(self, probs, input_length):
        pass


class GreedyDecoder(Decoder):
    """ Decode the best guess from probs using greedy algorithm """

    def decode(self, probs, input_length):
        # probs.shape = [batch_size, time_steps, num_classes]
        decoded = tf.keras.backend.ctc_decode(y_pred=probs,
                                              input_length=input_length,
                                              greedy=True)
        # decoded shape = [batch_size, decoded index]
        decoded = decoded[0]  # get the first result
        return self.convert_to_string(decoded)


class BeamSearchDecoder(Decoder):
    """ Decode probs using beam search algorithm """

    def __init__(self, index_to_token, beam_width=1024, lm_path=None):
        super().__init__(index_to_token)
        self.beam_width = beam_width
        self.lm_path = lm_path

    def decode(self, probs, input_length):
        # probs.shape = [batch_size, time_steps, num_classes]
        decoded = tf.keras.backend.ctc_decode(y_pred=probs,
                                              input_length=input_length,
                                              greedy=False,
                                              beam_width=self.beam_width)
        # decoded shape = [batch_size, top_path=1, decoded index]
        decoded = decoded[0]  # get the first object of the list of top-path objects
        return self.convert_to_string(decoded)


def create_decoder(name, index_to_token, beam_width=1024, lm_path=None):
    if name == "beamsearch":
        decoder = BeamSearchDecoder(index_to_token=index_to_token,
                                    beam_width=beam_width)
    elif name == "beamsearch_lm":
        if lm_path is None:
            raise ValueError("Missing 'lm_path' value in the configuration")
        decoder = BeamSearchDecoder(
            index_to_token=index_to_token,
            beam_width=beam_width,
            lm_path=lm_path)
    elif name == "greedy":
        decoder = GreedyDecoder(index_to_token=index_to_token)
    else:
        raise ValueError("'decoder' value must be either 'beamsearch',\
                         'beamsearch_lm' or 'greedy'")
    return decoder
