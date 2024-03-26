# pylint: disable=no-name-in-module,protected-access
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

"""
Causal padding supported Conv1D, Conv2D, DepthwiseConv1D, DepthwiseConv2D
"""

import importlib

import tensorflow as tf

from tensorflow_asr.utils.env_util import KERAS_SRC

Conv = importlib.import_module(f"{KERAS_SRC}.layers.convolutional.base_conv").Conv
conv_utils = importlib.import_module(f"{KERAS_SRC}.utils.conv_utils")


def _validate_init(self):  # removed check padding causal
    if self.filters is not None and self.filters % self.groups != 0:
        raise ValueError(
            f"The number of filters must be evenly divisible by the number of groups. Received: groups={self.groups}, filters={self.filters}"
        )
    if not all(self.kernel_size):
        raise ValueError(f"The argument `kernel_size` cannot contain 0(s). Received: {(self.kernel_size,)}")
    if not all(self.strides):
        raise ValueError(f"The argument `strides` cannot contains 0(s). Received: {(self.strides,)}")


def _compute_causal_padding(self, inputs):
    """Calculates padding for 'causal' option for 1-d and 2-d conv layers."""
    batch_pad = [[0, 0]]
    channel_pad = [[0, 0]]
    height_pad = [[self.dilation_rate[0] * (self.kernel_size[0] - 1), 0]]
    if self.rank == 1:
        if self.data_format == "channels_last":
            return batch_pad + height_pad + channel_pad
        return batch_pad + channel_pad + height_pad
    width_pad = [[self.dilation_rate[1] * (self.kernel_size[1] - 1), 0]]
    if self.data_format == "channels_last":
        return batch_pad + height_pad + width_pad + channel_pad
    return batch_pad + channel_pad + height_pad + width_pad


# Monkey patch
Conv._validate_init = _validate_init
Conv._compute_causal_padding = _compute_causal_padding

from keras.layers import Conv1D, Conv2D  # pylint: disable=unused-import
from keras.layers import DepthwiseConv1D as KerasDepthwiseConv1D
from keras.layers import DepthwiseConv2D as KerasDepthwiseConv2D


class DepthwiseConv1D(KerasDepthwiseConv1D):
    def __init__(
        self,
        kernel_size,
        strides=1,
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size,
            strides,
            padding,
            depth_multiplier,
            data_format,
            dilation_rate,
            activation,
            use_bias,
            depthwise_initializer,
            bias_initializer,
            depthwise_regularizer,
            bias_regularizer,
            activity_regularizer,
            depthwise_constraint,
            bias_constraint,
            **kwargs,
        )
        if self._is_causal:
            self.padding = "VALID"

    def call(self, inputs):
        if self._is_causal:
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            input_dim = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            input_dim = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        input_dim = conv_utils.conv_output_length(
            input_dim,
            self.kernel_size[0],
            "causal" if self._is_causal else self.padding,
            self.strides[0],
            self.dilation_rate[0],
        )
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, input_dim)
        if self.data_format == "channels_last":
            return (input_shape[0], input_dim, out_filters)
        raise ValueError("Invalid data_format")


class DepthwiseConv2D(KerasDepthwiseConv2D):
    def __init__(
        self,
        kernel_size,
        strides=...,
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=...,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size,
            strides,
            padding,
            depth_multiplier,
            data_format,
            dilation_rate,
            activation,
            use_bias,
            depthwise_initializer,
            bias_initializer,
            depthwise_regularizer,
            bias_regularizer,
            activity_regularizer,
            depthwise_constraint,
            bias_constraint,
            **kwargs,
        )
        if self._is_causal:
            self.padding = "VALID"

    def call(self, inputs):
        if self._is_causal:
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(
            rows,
            self.kernel_size[0],
            "causal" if self._is_causal else self.padding,
            self.strides[0],
            self.dilation_rate[0],
        )
        cols = conv_utils.conv_output_length(
            cols,
            self.kernel_size[1],
            "causal" if self._is_causal else self.padding,
            self.strides[1],
            self.dilation_rate[1],
        )
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, rows, cols)
        if self.data_format == "channels_last":
            return (input_shape[0], rows, cols, out_filters)
        raise ValueError("Invalid data_format")
