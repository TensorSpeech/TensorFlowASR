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
from tensorflow_asr.utils import shape_util


def compute_sinusoid_position_encoding(
    max_length,
    dmodel,
    input_length=None,
    interleave=False,
    direction="backward",
    dtype=tf.float32,
):
    assert direction in ["forward", "backward"]
    # length of sequence is the second last dimension of the inputs
    if direction == "forward":
        position = tf.cast(tf.range(0, max_length, 1), dtype=dtype)
        if input_length is not None:
            position *= tf.sequence_mask(input_length, max_length, dtype=dtype)
    else:
        position = tf.cast(tf.range(max_length - 1, -1, -1), dtype=dtype)
        if input_length is not None:
            position = tf.roll(position, shift=-(max_length - input_length), axis=0)
            position *= tf.sequence_mask(input_length, max_length, dtype=dtype)
    min_freq = tf.cast(1.0 / 10000.0, dtype=dtype)
    if interleave:
        timescales = tf.pow(min_freq, (2 * (tf.cast(tf.range(0, dmodel, 1.0), dtype=dtype) // 2)) / tf.cast(dmodel, dtype=dtype))
        angles = tf.einsum("i,d->id", position, timescales)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(0, dmodel, 1) % 2, dtype=dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
    else:
        timescales = tf.pow(min_freq, (tf.cast(tf.range(0, dmodel, 2.0), dtype=dtype) / tf.cast(dmodel, dtype=dtype)))
        angles = tf.einsum("i,d->id", position, timescales)
        positional_encodings = tf.concat([tf.sin(angles), tf.cos(angles)], -1)
    if input_length is not None:
        positional_encodings *= tf.sequence_mask(input_length, max_length, dtype=positional_encodings.dtype)[..., None]
    return positional_encodings


class SinusoidPositionalEncoding(Layer):
    def __init__(
        self,
        dynamic_encoding=True,
        interleave=False,
        direction="backward",
        **kwargs,
    ):
        super().__init__(trainable=False, **kwargs)
        self._dynamic_encoding = dynamic_encoding
        self._interleave = interleave
        self._direction = direction

    def call(self, inputs):
        outputs, outputs_length = inputs
        batch_size, max_length, dmodel = shape_util.shape_list(outputs)

        if not self._dynamic_encoding:
            pe = compute_sinusoid_position_encoding(
                max_length, dmodel, input_length=None, interleave=self._interleave, direction=self._direction, dtype=outputs.dtype
            )
            pe = tf.repeat(pe[None, :, :], repeats=batch_size, axis=0)
            pe = tf.stop_gradient(pe)
            return pe

        def _fn(input_length):
            return compute_sinusoid_position_encoding(
                max_length, dmodel, input_length=input_length, interleave=self._interleave, direction=self._direction, dtype=outputs.dtype
            )

        pe = tf.map_fn(
            fn=_fn,
            elems=outputs_length,
            fn_output_signature=tf.TensorSpec(shape=outputs.shape.as_list()[1:], dtype=outputs.dtype),
        )
        pe = tf.stop_gradient(pe)
        return pe

    def compute_output_shape(self, input_shape):
        output_shape, _ = input_shape
        return tuple(output_shape)
