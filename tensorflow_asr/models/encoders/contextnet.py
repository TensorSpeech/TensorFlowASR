# Copyright 2020 Huy Le Nguyen (@nglehuy)
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
""" http://arxiv.org/abs/2005.03191 """

from typing import List

import tensorflow as tf

from tensorflow_asr.models.base_layer import Layer, Reshape
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


class ConvModule(Layer):
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
            dtype=self.dtype,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.activation = get_activation(activation)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.conv(outputs, training=training)
        outputs_length = math_util.conv_output_length(
            outputs_length, filter_size=self.conv.kernel_size[0], padding=self.conv.padding, stride=self.conv.strides[0]
        )
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (
            math_util.conv_output_length(length, filter_size=self.conv.kernel_size[0], padding=self.conv.padding, stride=self.conv.strides[0])
            for length in (maxlen, outputs_length)
        )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.conv.compute_output_shape(output_shape)
        return output_shape, output_length_shape


class SEModule(Layer):
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
            dtype=self.dtype,
        )
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(keepdims=True, name="global_avg_pool", dtype=self.dtype)
        self.activation = get_activation(activation)
        self.fc1 = tf.keras.layers.Dense(
            filters // 8,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="fc1",
            dtype=self.dtype,
        )
        self.fc2 = tf.keras.layers.Dense(
            filters,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="fc2",
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv((outputs, outputs_length), training=training)  # [B, T, E]

        se = self.global_avg_pool(outputs)  # [B, 1, E], mask auto populate
        se = self.fc1(se, training=training)
        se = self.activation(se)
        se = self.fc2(se, training=training)
        se = tf.nn.sigmoid(se)

        se = tf.tile(se, [1, tf.shape(outputs)[1], 1])  # [B, 1, E] => [B, T, E]
        outputs = tf.multiply(outputs, se)  # [B, T, E]
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        return self.conv.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class ConvBlock(Layer):
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

        self.time_reduction_factor = strides
        filters = int(filters * alpha)
        self.dmodel = filters

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
                    dtype=self.dtype,
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
            dtype=self.dtype,
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
            dtype=self.dtype,
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
                dtype=self.dtype,
            )

        self.activation = get_activation(activation)

    def call(self, inputs, training=False):
        _inputs, _inputs_length = inputs
        outputs, outputs_length = _inputs, _inputs_length
        for conv in self.convs:
            outputs, outputs_length = conv((outputs, outputs_length), training=training)
        outputs, outputs_length = self.last_conv((outputs, outputs_length), training=training)
        outputs, outputs_length = self.se((outputs, outputs_length), training=training)
        if self.residual is not None:
            res, _ = self.residual((_inputs, _inputs_length), training=training)
            outputs = tf.add(outputs, res)
        outputs = self.activation(outputs)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        return self.last_conv.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for conv in self.convs:
            output_shape = conv.compute_output_shape(output_shape)
        output_shape = self.last_conv.compute_output_shape(output_shape)
        output_shape = self.se.compute_output_shape(output_shape)
        return output_shape


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

        self.reshape = Reshape(name="reshape", dtype=self.dtype)

        self.blocks = []
        self.time_reduction_factor = 1
        for i, config in enumerate(blocks):
            block = ConvBlock(
                **config,
                alpha=alpha,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
                dtype=self.dtype,
            )
            self.blocks.append(block)
            self.time_reduction_factor *= block.time_reduction_factor

        self.dmodel = self.blocks[-1].dmodel

    def call(self, inputs, training=False):
        outputs, outputs_length, caching = inputs
        outputs, outputs_length = self.reshape((outputs, outputs_length))
        for block in self.blocks:
            outputs, outputs_length = block((outputs, outputs_length), training=training)
        return outputs, outputs_length, caching

    def call_next(self, features, features_length, *args, **kwargs):
        """
        Recognize function for encoder network

        Parameters
        ----------
        features : tf.Tensor, shape [B, T, F, C]
        features_length : tf.Tensor, shape [B]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor], shape ([B, T, dmodel], [B], None)
            Outputs, outputs_length, new_states
        """
        with tf.name_scope(f"{self.name}_call_next"):
            outputs, outputs_length, _ = self.call((features, features_length, None), training=False)
            return outputs, outputs_length, None

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length, caching = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (math_util.get_reduced_length(length, self.time_reduction_factor) for length in (maxlen, outputs_length))
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None, getattr(caching, "_keras_mask", None)

    def compute_output_shape(self, input_shape):
        *output_shape, caching_shape = input_shape
        output_shape = self.reshape.compute_output_shape(output_shape)
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return *output_shape, caching_shape
