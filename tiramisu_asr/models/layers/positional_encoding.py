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
from ...utils.utils import shape_list


def positional_encoding(max_len, size):
    pos = tf.expand_dims(tf.range(0, max_len, dtype=tf.float32), axis=1)
    index = tf.expand_dims(tf.range(0, size, dtype=tf.float32), axis=0)

    pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2)) / size))

    # Sin cos will be [max_len, size // 2], we add 0 between number by using padding and reshape
    sin = tf.pad(tf.expand_dims(tf.sin(pe[:, 0::2]), -1), [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
    sin = tf.reshape(sin, [max_len, size])
    cos = tf.pad(tf.expand_dims(tf.cos(pe[:, 1::2]), -1), [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
    cos = tf.reshape(cos, [max_len, size])
    # Then add sin and cos
    pe = tf.add(sin, cos)

    return tf.expand_dims(pe, axis=0)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 name="pos_enc",
                 **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_sequence_length, size = shape_list(inputs)
        if size % 2 != 0: raise ValueError(f"Input last dim must be even: {size}")
        pe = positional_encoding(max_sequence_length, size)
        return inputs + tf.cast(pe, dtype=inputs.dtype)

    def get_config(self):
        conf = super(PositionalEncoding, self).get_config()
        return conf
