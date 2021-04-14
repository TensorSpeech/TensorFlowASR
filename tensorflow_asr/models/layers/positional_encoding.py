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
from ...utils.shape_util import shape_list


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, alpha: int = 1, beta: int = 0, name="positional_encoding", **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def build(self, input_shape):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(max_len, dmodel):
        pos = tf.expand_dims(tf.range(max_len - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2)) / dmodel))

        # Sin cos will be [max_len, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(tf.expand_dims(tf.sin(pe[:, 0::2]), -1),
                     [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
        sin = tf.reshape(sin, [max_len, dmodel])
        cos = tf.pad(tf.expand_dims(tf.cos(pe[:, 1::2]), -1),
                     [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
        cos = tf.reshape(cos, [max_len, dmodel])
        # Then add sin and cos, which results in [time, size]
        pe = tf.add(sin, cos)
        return tf.expand_dims(pe, axis=0)  # [1, time, size]

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len * self.alpha + self.beta, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)

    def get_config(self):
        conf = super().get_config()
        return conf.update({"alpha": self.alpha, "beta": self.beta})


class PositionalEncodingConcat(PositionalEncoding):
    def build(self, input_shape):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(max_len, dmodel):
        pos = tf.range(max_len - 1, -1, -1.0, dtype=tf.float32)

        index = tf.range(0, dmodel, 2.0, dtype=tf.float32)
        index = 1 / tf.pow(10000.0, (index / dmodel))

        sinusoid = tf.einsum("i,j->ij", pos, index)
        pos = tf.concat([tf.sin(sinusoid), tf.cos(sinusoid)], axis=-1)

        return tf.expand_dims(pos, axis=0)

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len * self.alpha + self.beta, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)
