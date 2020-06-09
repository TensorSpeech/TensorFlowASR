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
import tensorflow as tf
from nltk.metrics import distance
from .utils import bytes_to_string


def wer(decode: tf.Tensor, target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    decode = bytes_to_string(decode.numpy())
    target = bytes_to_string(target.numpy())
    dis = 0.0
    length = 0.0
    for dec, tar in zip(decode, target):
        words = set(dec.split() + tar.split())
        word2char = dict(zip(words, range(len(words))))

        new_decode = [chr(word2char[w]) for w in dec.split()]
        new_target = [chr(word2char[w]) for w in tar.split()]

        dis += distance.edit_distance(''.join(new_decode), ''.join(new_target))
        length += len(tar.split())
    return tf.convert_to_tensor(dis, tf.float32), tf.convert_to_tensor(length, tf.float32)


def cer(decode: tf.Tensor, target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    decode = bytes_to_string(decode.numpy())
    target = bytes_to_string(target.numpy())
    dis = 0
    length = 0
    for dec, tar in zip(decode, target):
        dis += distance.edit_distance(dec, tar)
        length += len(tar)
    return tf.convert_to_tensor(dis, tf.float32), tf.convert_to_tensor(length, tf.float32)


class ErrorRate(tf.keras.metrics.Metric):
    """ Metric for WER and CER """

    def __init__(self, func, name="error_rate", **kwargs):
        super(ErrorRate, self).__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name=f"{name}_numerator", initializer="zeros")
        self.denominator = self.add_weight(name=f"{name}_denominator", initializer="zeros")
        self.func = func

    def update_state(self, decode, target):
        n, d = tf.py_function(self.func, inp=[decode, target], Tout=[tf.float32, tf.float32])
        self.numerator.assign_add(n)
        self.denominator.assign_add(d)

    def result(self):
        if self.denominator == 0: return 0.0
        return (self.numerator / self.denominator) * 100
