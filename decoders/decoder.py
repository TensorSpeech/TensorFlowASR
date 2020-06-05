# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import

import abc
import multiprocessing
import tensorflow as tf
from featurizers.text_featurizers import TextFeaturizer


class Decoder(metaclass=abc.ABCMeta):
    def __init__(self,
                 decoder_config: dict,
                 text_featurizer: TextFeaturizer):
        self.num_cpus = multiprocessing.cpu_count()
        self.decoder_config = decoder_config
        self.text_featurizer = text_featurizer
        self.func = self.set_decoding_func()

    @tf.function
    def convert_to_string(self, batch: tf.Tensor) -> tf.Tensor:
        # Convert to string
        _map = lambda x: tf.py_function(self.text_featurizer.iextract, inp=[x], Tout=tf.string)
        return tf.map_fn(_map, batch, dtype=tf.string)

    @tf.function
    def decode(self, probs, input_length):
        probs = tf.nn.softmax(probs)
        return tf.py_function(self.func, inp=[probs, input_length], Tout=tf.string)

    @abc.abstractmethod
    def set_decoding_func(self):
        """ Set function to perform decoding """
        pass
