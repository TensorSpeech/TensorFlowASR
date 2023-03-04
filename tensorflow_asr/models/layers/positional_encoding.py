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

from tensorflow_asr.models.layers.base_layer import Layer
from tensorflow_asr.utils import math_util, shape_util


def compute_sinusoid_position_encoding(
    input_length,
    max_length,
    dmodel,
    dtype=tf.float32,
):
    # length of sequence is the second last dimension of the inputs
    position = tf.cast(tf.range(max_length - 1, -1, -1), dtype=dtype)
    position = tf.roll(position, shift=-(max_length - input_length), axis=0)
    position *= tf.sequence_mask(input_length, max_length, dtype=dtype)
    min_freq = tf.cast(1.0 / 10000.0, dtype=dtype)
    timescales = tf.pow(min_freq, ((tf.cast(tf.range(0, dmodel, 1), dtype=dtype) // 2) * 2) / tf.cast(dmodel, dtype=dtype))
    angles = tf.einsum("i,d->id", position, timescales)
    # even indices are sine, odd are cosine
    cos_mask = tf.cast(tf.range(0, dmodel, 1), dtype=dtype) % 2
    sin_mask = 1 - cos_mask
    # embedding shape is [seq_length, hidden_size]
    positional_encodings = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
    return positional_encodings
    # return tf.tile(positional_encodings[None, :, :], [batch_size, 1, 1])


class SinusoidPositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def call(self, inputs):
        outputs, outputs_length = inputs
        _, max_length, dmodel = shape_util.shape_list(outputs)

        def _fn(input_length):
            return compute_sinusoid_position_encoding(input_length, max_length, dmodel, dtype=outputs.dtype)

        pe = tf.map_fn(
            fn=_fn,
            elems=outputs_length,
            fn_output_signature=tf.TensorSpec(shape=outputs.shape.as_list()[1:], dtype=outputs.dtype),
        )
        mask = getattr(outputs, "_keras_mask", None)
        if mask is not None:
            pe = math_util.apply_mask(pe, mask=mask)

        return pe

    def compute_output_shape(self, input_shape):
        output_shape, _ = input_shape
        return tuple(output_shape)
