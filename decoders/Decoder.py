from __future__ import absolute_import

import multiprocessing
import tensorflow as tf
import numpy as np
from utils.Utils import bytes_to_string


class Decoder:
    def __init__(self, text_featurizer):
        # tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
        # default blank index is -1
        self.blank_index = -1
        self.num_cpus = multiprocessing.cpu_count()
        self.text_featurizer = text_featurizer

    def convert_to_string(self, decoded) -> np.ndarray:
        # Remove blank indices
        def map_cvrt(elem):
            elem = elem.numpy()
            elem = elem[elem != self.blank_index]
            elem = elem[elem != self.text_featurizer.num_classes - 1]
            return ''.join([self.text_featurizer.index_to_token[i] for i in elem])

        # Convert to string
        decoded = tf.map_fn(map_cvrt, decoded, dtype=tf.string)
        return bytes_to_string(decoded.numpy())

    def convert_to_string_single(self, decoded: np.ndarray) -> str:
        decoded = decoded[decoded != self.blank_index]
        decoded = decoded[decoded != self.text_featurizer.num_classes - 1]
        return ''.join([self.text_featurizer.index_to_token[i] for i in decoded])

    def string_to_index(self, decoded: np.ndarray) -> tf.Tensor:
        def map_cvrt(elem):
            result = []
            for char in elem:
                result.append(self.text_featurizer.token_to_index[char])
            return result

        return tf.convert_to_tensor(map_cvrt(decoded), dtype=tf.int32)

    def __call__(self, probs, input_length, last_activation="linear"):
        if last_activation not in ["softmax"]:
            probs = tf.nn.softmax(probs)
        return tf.py_function(self.decode, inp=[probs, input_length],
                              Tout=tf.string)

    def decode(self, probs, input_length):
        pass
