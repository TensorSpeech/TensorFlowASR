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
""" Ref: https://github.com/iankur/ContextNet """

from typing import List
import tensorflow as tf
from ...utils import math_util

L2 = tf.keras.regularizers.l2(1e-6)


def get_activation(activation: str = "silu"):
    activation = activation.lower()
    if activation in ["silu", "swish"]: return tf.nn.swish
    elif activation == "relu": return tf.nn.relu
    elif activation == "linear": return tf.keras.activations.linear
    else: raise ValueError("activation must be either 'silu', 'swish', 'relu' or 'linear'")


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs): return math_util.merge_two_last_dims(inputs)


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ConvModule, self).__init__(**kwargs)
        self.strides = strides
        self.conv = tf.keras.layers.SeparableConv1D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
            depthwise_regularizer=kernel_regularizer, pointwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, name=f"{self.name}_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.activation = get_activation(activation)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs


class SEModule(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(SEModule, self).__init__(**kwargs)
        self.conv = ConvModule(
            kernel_size=kernel_size, strides=strides,
            filters=filters, activation=activation,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv_module"
        )
        self.activation = get_activation(activation)
        self.fc1 = tf.keras.layers.Dense(filters // 8, name=f"{self.name}_fc1")
        self.fc2 = tf.keras.layers.Dense(filters, name=f"{self.name}_fc2")

    def call(self, inputs, training=False, **kwargs):
        features, input_length = inputs
        outputs = self.conv(features, training=training)

        se = tf.divide(tf.reduce_sum(outputs, axis=1), tf.expand_dims(tf.cast(input_length, dtype=outputs.dtype), axis=1))
        se = self.fc1(se, training=training)
        se = self.activation(se)
        se = self.fc2(se, training=training)
        se = self.activation(se)
        se = tf.nn.sigmoid(se)
        se = tf.expand_dims(se, axis=1)

        outputs = tf.multiply(outputs, se)
        return outputs


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 nlayers: int = 3,
                 kernel_size: int = 3,
                 filters: int = 256,
                 strides: int = 1,
                 residual: bool = True,
                 activation: str = 'silu',
                 alpha: float = 1.0,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.dmodel = filters
        self.time_reduction_factor = strides
        filters = int(filters * alpha)

        self.convs = []
        for i in range(nlayers - 1):
            self.convs.append(
                ConvModule(
                    kernel_size=kernel_size, strides=1,
                    filters=filters, activation=activation,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    name=f"{self.name}_conv_module_{i}"
                )
            )

        self.last_conv = ConvModule(
            kernel_size=kernel_size, strides=strides,
            filters=filters, activation=activation,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv_module_{nlayers - 1}"
        )

        self.se = SEModule(
            kernel_size=kernel_size, strides=1, filters=filters, activation=activation,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=f"{self.name}_se"
        )

        self.residual = None
        if residual:
            self.residual = ConvModule(
                kernel_size=kernel_size, strides=strides,
                filters=filters, activation="linear",
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                name=f"{self.name}_residual"
            )

        self.activation = get_activation(activation)

    def call(self, inputs, training=False, **kwargs):
        features, input_length = inputs
        outputs = features
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.last_conv(outputs, training=training)
        input_length = math_util.get_reduced_length(input_length, self.last_conv.strides)
        outputs = self.se([outputs, input_length], training=training)
        if self.residual is not None:
            res = self.residual(features, training=training)
            outputs = tf.add(outputs, res)
        outputs = self.activation(outputs)
        return outputs, input_length


class ContextNetEncoder(tf.keras.Model):
    def __init__(self,
                 blocks: List[dict] = [],
                 alpha: float = 1.0,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ContextNetEncoder, self).__init__(**kwargs)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.blocks = []
        for i, config in enumerate(blocks):
            self.blocks.append(
                ConvBlock(
                    **config, alpha=alpha,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    name=f"{self.name}_block_{i}"
                )
            )

    def call(self, inputs, training=False, **kwargs):
        outputs, input_length = inputs
        outputs = self.reshape(outputs)
        for block in self.blocks:
            outputs, input_length = block([outputs, input_length], training=training)
        return outputs
