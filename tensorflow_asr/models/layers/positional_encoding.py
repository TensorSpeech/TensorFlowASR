# pylint: disable=attribute-defined-outside-init
# Copyright 2022 Huy Le Nguyen (@usimarit)
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


def compute_sinusoid_position_encoding(
    batch_size,
    max_length,
    dmodel,
    dtype=tf.float32,
):
    # length of sequence is the second last dimension of the inputs
    position = tf.cast(tf.range(0, max_length, 1), dtype)
    min_freq = tf.cast(1 / 10000.0, dtype=dtype)
    timescales = tf.pow(min_freq, tf.cast(2 * (tf.range(dmodel) // 2), dtype) / tf.cast(dmodel, dtype))
    angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
    # even indices are sine, odd are cosine
    cos_mask = tf.cast(tf.range(dmodel) % 2, dtype)
    sin_mask = 1 - cos_mask
    # embedding shape is [seq_length, hidden_size]
    positional_encodings = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
    return tf.tile(positional_encodings[None, :, :], [batch_size, 1, 1])
