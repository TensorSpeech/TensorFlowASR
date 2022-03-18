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

from typing import List

import tensorflow as tf

from tensorflow_asr.models.layers.base_layer import Layer
from tensorflow_asr.utils import math_util

L2 = tf.keras.regularizers.l2(1e-6)


def get_activation(
    activation: str = "silu",
):
    activation = activation.lower()
    if activation in ["silu", "swish"]:
        return tf.nn.swish
    if activation == "relu":
        return tf.nn.relu
    if activation == "linear":
        return tf.keras.activations.linear
    raise ValueError("activation must be either 'silu', 'swish', 'relu' or 'linear'")


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs):
        return math_util.merge_two_last_dims(inputs)


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size: int = 3,
        strides: int = 1,
        filters: int = 256,
        activation: str = "silu",
        padding: str = "causal",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strides = strides
        self.conv = tf.keras.layers.SeparableConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depthwise_regularizer=kernel_regularizer,
            pointwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn")
        self.activation = get_activation(activation)

    def call(self, inputs, training=False):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs


class SEModule(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size: int = 3,
        strides: int = 1,
        filters: int = 256,
        activation: str = "silu",
        padding: str = "causal",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = ConvModule(
            kernel_size=kernel_size,
            strides=strides,
            filters=filters,
            activation=activation,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="conv_module",
        )
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(keepdims=True, name="global_avg_pool")
        self.activation = get_activation(activation)
        self.fc1 = tf.keras.layers.Dense(filters // 8, name="fc1")
        self.fc2 = tf.keras.layers.Dense(filters, name="fc2")

    def call(self, inputs, training=False):
        features, inputs_length = inputs
        outputs = self.conv(features, training=training)  # [B, T, E]

        mask = tf.sequence_mask(inputs_length, maxlen=tf.shape(outputs)[1])
        se = self.global_avg_pool(outputs, mask=mask)  # [B, 1, E]
        se = self.fc1(se, training=training)
        se = self.activation(se)
        se = self.fc2(se, training=training)
        se = tf.nn.sigmoid(se)

        se = tf.tile(se, [1, tf.shape(outputs)[1], 1])  # [B, 1, E] => [B, T, E]
        outputs = tf.multiply(outputs, se)  # [B, T, E]
        return outputs


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nlayers: int = 3,
        kernel_size: int = 3,
        filters: int = 256,
        strides: int = 1,
        residual: bool = True,
        activation: str = "silu",
        alpha: float = 1.0,
        padding: str = "causal",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dmodel = filters
        self.time_reduction_factor = strides
        filters = int(filters * alpha)

        self.convs = []
        for i in range(nlayers - 1):
            self.convs.append(
                ConvModule(
                    kernel_size=kernel_size,
                    strides=1,
                    filters=filters,
                    activation=activation,
                    padding=padding,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f"conv_module_{i}",
                )
            )

        self.last_conv = ConvModule(
            kernel_size=kernel_size,
            strides=strides,
            filters=filters,
            activation=activation,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"conv_module_{nlayers - 1}",
        )

        self.se = SEModule(
            kernel_size=kernel_size,
            strides=1,
            filters=filters,
            activation=activation,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="se",
        )

        self.residual = None
        if residual:
            self.residual = ConvModule(
                kernel_size=kernel_size,
                strides=strides,
                filters=filters,
                activation="linear",
                padding=padding,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="residual",
            )

        self.activation = get_activation(activation)

    def call(self, inputs, training=False):
        features, inputs_length = inputs
        outputs = features
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.last_conv(outputs, training=training)
        inputs_length = math_util.get_reduced_length(inputs_length, self.last_conv.strides)
        outputs = self.se([outputs, inputs_length], training=training)
        if self.residual is not None:
            res = self.residual(features, training=training)
            outputs = tf.add(outputs, res)
        outputs = self.activation(outputs)
        return outputs, inputs_length


class ContextNetEncoder(Layer):
    def __init__(
        self,
        blocks: List[dict] = [],
        alpha: float = 1.0,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reshape = Reshape(name="reshape")

        self.blocks = []
        for i, config in enumerate(blocks):
            self.blocks.append(
                ConvBlock(
                    **config,
                    alpha=alpha,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f"block_{i}",
                )
            )

        self.dmodel = self.blocks[-1].dmodel
        self.time_reduction_factor = 1
        for block in self.blocks:
            self.time_reduction_factor *= block.time_reduction_factor

    def call(self, inputs, training=False):
        outputs, inputs_length = inputs
        outputs = self.reshape(outputs)
        for block in self.blocks:
            outputs, inputs_length = block([outputs, inputs_length], training=training)
        return outputs, inputs_length

    def compute_output_shape(self, input_shape):
        inputs_shape, inputs_length_shape = input_shape
        outputs_size = self.dmodel
        outputs_time = None if inputs_shape[1] is None else math_util.legacy_get_reduced_length(inputs_shape[1], self.time_reduction_factor)
        outputs_batch = inputs_shape[0]
        outputs_shape = [outputs_batch, outputs_time, outputs_size]
        return tuple(outputs_shape), tuple(inputs_length_shape)
