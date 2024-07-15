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

# import importlib

from keras.src.ops.operation_utils import compute_conv_output_shape

from tensorflow_asr import keras, tf

# from tensorflow_asr.utils.env_util import KERAS_SRC

# Conv = importlib.import_module(f"{KERAS_SRC}.layers.convolutional.base_conv").Conv
# conv_utils = importlib.import_module(f"{KERAS_SRC}.utils.conv_utils")


# def _validate_init(self):  # removed check padding causal
#     if self.filters is not None and self.filters % self.groups != 0:
#         raise ValueError(
#             f"The number of filters must be evenly divisible by the number of groups. Received: groups={self.groups}, filters={self.filters}"
#         )
#     if not all(self.kernel_size):
#         raise ValueError(f"The argument `kernel_size` cannot contain 0(s). Received: {(self.kernel_size,)}")
#     if not all(self.strides):
#         raise ValueError(f"The argument `strides` cannot contains 0(s). Received: {(self.strides,)}")


def _compute_causal_padding(inputs, rank, data_format, dilation_rate, kernel_size):
    """Calculates padding for 'causal' option for 1-d and 2-d conv layers."""
    batch_pad = [[0, 0]]
    channel_pad = [[0, 0]]
    height_pad = [[dilation_rate[0] * (kernel_size[0] - 1), 0]]
    if rank == 1:
        if data_format == "channels_last":
            return batch_pad + height_pad + channel_pad
        return batch_pad + channel_pad + height_pad
    width_pad = [[dilation_rate[1] * (kernel_size[1] - 1), 0]]
    if data_format == "channels_last":
        return batch_pad + height_pad + width_pad + channel_pad
    return batch_pad + channel_pad + height_pad + width_pad


@keras.utils.register_keras_serializable(package=__name__)
class Conv2D(keras.layers.Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        if padding == "causal":
            self._is_causal = True
            padding = "valid"
        else:
            self._is_causal = False
        super().__init__(
            filters,
            kernel_size,
            strides,
            padding,
            data_format,
            dilation_rate,
            groups,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )

    def call(self, inputs):
        if self._is_causal:
            inputs = tf.pad(
                inputs,
                _compute_causal_padding(
                    inputs,
                    rank=self.rank,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                    kernel_size=self.kernel_size,
                ),
            )
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding="causal" if self._is_causal else self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


@keras.utils.register_keras_serializable(package=__name__)
class DepthwiseConv1D(keras.layers.DepthwiseConv1D):
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
        if padding == "causal":
            self._is_causal = True
            padding = "valid"
        else:
            self._is_causal = False
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

    def call(self, inputs):
        if self._is_causal:
            inputs = tf.pad(
                inputs,
                _compute_causal_padding(
                    inputs,
                    rank=self.rank,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                    kernel_size=self.kernel_size,
                ),
            )
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        input_channel = self._get_input_channel(input_shape)
        return compute_conv_output_shape(
            input_shape,
            self.depth_multiplier * input_channel,
            self.kernel_size,
            strides=self.strides,
            padding="causal" if self._is_causal else self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


@keras.utils.register_keras_serializable(package=__name__)
class DepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
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
        if padding == "causal":
            self._is_causal = True
            padding = "valid"
        else:
            self._is_causal = False
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

    def call(self, inputs):
        if self._is_causal:
            inputs = tf.pad(
                inputs,
                _compute_causal_padding(
                    inputs,
                    rank=self.rank,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                    kernel_size=self.kernel_size,
                ),
            )
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        input_channel = self._get_input_channel(input_shape)
        return compute_conv_output_shape(
            input_shape,
            self.depth_multiplier * input_channel,
            self.kernel_size,
            strides=self.strides,
            padding="causal" if self._is_causal else self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
