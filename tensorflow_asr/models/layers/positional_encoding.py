# pylint: disable=attribute-defined-outside-init
# Copyright 2022 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.utils import shape_util


def compute_position(
    start,
    end,
    step,
    dtype=tf.float32,
):
    return tf.cast(tf.range(start, end, step), dtype=dtype)


def compute_sinusoid_position_encoding(
    position,
    batch_size,
    dmodel,
    interleave=False,
    dtype=tf.float32,
):
    min_freq = tf.cast(1.0 / 10000.0, dtype=dtype)
    if interleave:
        timescales = tf.pow(min_freq, (2 * (tf.cast(tf.range(0, dmodel, 1.0), dtype=dtype) // 2)) / tf.cast(dmodel, dtype=dtype))
        angles = tf.einsum("i,d->id", position, timescales)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(0, dmodel, 1) % 2, dtype=dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        pe = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
    else:
        timescales = tf.pow(min_freq, (tf.cast(tf.range(0, dmodel, 2.0), dtype=dtype) / tf.cast(dmodel, dtype=dtype)))
        angles = tf.einsum("i,d->id", position, timescales)
        pe = tf.concat([tf.sin(angles), tf.cos(angles)], -1)
    pe = tf.repeat(pe[None, :, :], repeats=batch_size, axis=0)
    return pe


class SinusoidalPositionalEncoding(Layer):
    def __init__(
        self,
        dropout=0,
        scale=None,
        interleave=False,
        **kwargs,
    ):
        super().__init__(trainable=False, **kwargs)
        self.do = tf.keras.layers.Dropout(dropout, dtype=self.dtype, name="dropout")
        self._scale = scale
        self._interleave = interleave

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        if self._scale is not None:
            outputs *= self._scale
        batch_size, length, dmodel = shape_util.shape_list(outputs)
        position = compute_position(start=0, end=length, step=1, dtype=outputs.dtype)
        pe = compute_sinusoid_position_encoding(
            position=position,
            batch_size=batch_size,
            dmodel=dmodel,
            interleave=self._interleave,
            dtype=outputs.dtype,
        )
        pe *= tf.sequence_mask(outputs_length, maxlen=length, dtype=pe.dtype)
        pe = self.do(pe, training=training)
        outputs += pe
        return outputs, pe

    def compute_output_shape(self, input_shape):
        output_shape, _ = input_shape
        return output_shape, output_shape


class RelativeSinusoidalPositionalEncoding(SinusoidalPositionalEncoding):
    def __init__(
        self,
        dropout=0,
        scale=None,
        interleave=False,
        memory_length=None,
        causal=False,
        **kwargs,
    ):
        """
        http://arxiv.org/abs/1901.02860
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
        Relative Sinusoidal Positional Encoding
        Will be computed with weights as the Q in paper
        ==> Define in reversed order
        """
        super().__init__(dropout, scale, interleave, **kwargs)
        self._memory_length = memory_length or 0
        self._causal = causal

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        if self._scale is not None:
            outputs *= self._scale
        batch_size, length, dmodel = shape_util.shape_list(outputs)
        position_left = compute_position(start=length + self._memory_length - 1, end=0, step=-1, dtype=outputs.dtype)
        position_right = compute_position(start=0, end=-length, step=-1, dtype=outputs.dtype)
        position = tf.concat([position_left, position_right], axis=0)  # 2 * length + self._memory_length - 1
        pe = compute_sinusoid_position_encoding(
            position=position,
            batch_size=batch_size,
            dmodel=dmodel,
            interleave=self._interleave,
            dtype=outputs.dtype,
        )
        if self._causal:
            pe, _ = tf.map_fn(
                fn=lambda x: (  # [B, length + self._memory_length, dmodel]
                    tf.multiply(
                        tf.slice(
                            tf.roll(input=x[0], shift=-(length - x[1]), axis=0),
                            begin=[0, 0],
                            size=[(length + self._memory_length), dmodel],
                        ),
                        tf.expand_dims(
                            tf.sequence_mask((x[1] + self._memory_length), maxlen=(length + self._memory_length), dtype=x[0].dtype),
                            axis=-1,
                        ),
                    ),
                    x[1],
                ),
                elems=(pe, outputs_length),
                # fn_output_signature=(
                #     tf.TensorSpec(shape=[(length + self._memory_length), dmodel], dtype=pe.dtype),
                #     tf.TensorSpec(shape=[], dtype=outputs_length.dtype),
                # ),
            )
        else:
            pe, _ = tf.map_fn(
                fn=lambda x: (  # [B, 2 * length + self._memory_length - 1, dmodel]
                    tf.multiply(
                        tf.slice(
                            tf.roll(input=x[0], shift=-(length - x[1]), axis=0),
                            begin=[0, 0],
                            size=[(2 * length + self._memory_length - 1), dmodel],
                        ),
                        tf.expand_dims(
                            tf.sequence_mask((2 * x[1] + self._memory_length - 1), maxlen=(2 * length + self._memory_length - 1), dtype=x[0].dtype),
                            axis=-1,
                        ),
                    ),
                    x[1],
                ),
                elems=(pe, outputs_length),
                # fn_output_signature=(
                #     tf.TensorSpec(shape=[(2 * length + self._memory_length - 1), dmodel], dtype=pe.dtype),
                #     tf.TensorSpec(shape=[], dtype=outputs_length.dtype),
                # ),
            )
        pe = self.do(pe, training=training)
        return outputs, pe

    def compute_output_shape(self, input_shape):
        output_shape, _ = input_shape
        B, T, V = output_shape
        pT = 2 * T - 1 if T is not None else None
        if self._memory_length > 0 and T is not None:
            pT += self._memory_length
        return output_shape, (B, pT, V)
