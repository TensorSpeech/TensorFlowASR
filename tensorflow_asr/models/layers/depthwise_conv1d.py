# pylint: disable=no-name-in-module
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
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops


class DepthwiseConv1D(tf.keras.layers.DepthwiseConv1D):
    """
    Causal padding supported DepthwiseConv1D
    """

    def _validate_init(self):  # removed check padding causal
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the number of groups. "
                f"Received: groups={self.groups}, filters={self.filters}"
            )
        if not all(self.kernel_size):
            raise ValueError(f"The argument `kernel_size` cannot contain 0(s). Received: {(self.kernel_size,)}")
        if not all(self.strides):
            raise ValueError(f"The argument `strides` cannot contains 0(s). Received: {(self.strides,)}")

    def call(self, inputs):
        # input will be in shape [B, T, E] for channel_last or [B, E, T] for channel_first
        if self._is_causal:
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        if self.data_format == "channels_last":  # default
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1  # [B, 1, T, E]
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2  # [B, E, 1, T]
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)  # (1, kernel_size) across T dimension
        dilation_rate = (1,) + self.dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=self.padding.upper() if not self._is_causal else "VALID",
            dilations=dilation_rate,
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        outputs = tf.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
