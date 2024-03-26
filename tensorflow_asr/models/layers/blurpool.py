# Adopted from https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
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

import numpy as np
import tensorflow as tf

from tensorflow_asr.models.base_layer import Layer


class BlurPool2D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 4,
        strides: int = 2,
        padding: str = "reflect",
        trainable=True,
        name="blurpool2d",
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad_mode = "CONSTANT" if padding == "valid" else padding.upper()
        if padding != "valid":
            self.paddings = [
                [0, 0],  # batch
                [int(1.0 * (kernel_size - 1) / 2), int(np.ceil(1.0 * (kernel_size - 1) / 2))],  # height
                [int(1.0 * (kernel_size - 1) / 2), int(np.ceil(1.0 * (kernel_size - 1) / 2))],  # width
                [0, 0],  # channel
            ]
        else:
            self.paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]

        if self.kernel_size == 1:
            a = np.array([1.0])
        elif self.kernel_size == 2:
            a = np.array([1.0, 1.0])
        elif self.kernel_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.kernel_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.kernel_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.kernel_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.kernel_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        self.kernel = tf.constant(a[:, None] * a[None, :], dtype=self.compute_dtype)
        self.kernel = tf.divide(self.kernel, tf.reduce_sum(self.kernel))
        self.kernel = tf.repeat(tf.expand_dims(self.kernel, -1), self.filters, axis=-1)  # [kernel_size, kernel_size, filters]

    def call(self, inputs):
        inputs = tf.pad(inputs, paddings=self.paddings, mode=self.pad_mode)
        in_channels = tf.shape(inputs)[3]
        # [kernel_size, kernel_size, in_channel, filters]
        kernel = tf.repeat(tf.expand_dims(self.kernel, axis=2), in_channels, axis=2)
        return tf.nn.conv2d(inputs, filters=kernel, strides=self.strides, padding="VALID")


class BlurPool1D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 4,
        strides: int = 2,
        padding: str = "reflect",
        trainable=True,
        name="blurpool1d",
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad_mode = "CONSTANT" if padding == "valid" else padding.upper()
        if padding != "valid":
            self.paddings = [
                [0, 0],  # batch
                [int(1.0 * (kernel_size - 1) / 2), int(np.ceil(1.0 * (kernel_size - 1) / 2))],  # width
                [0, 0],  # channel
            ]
        else:
            self.paddings = [[0, 0], [0, 0], [0, 0]]

        if self.kernel_size == 1:
            a = np.array([1.0])
        elif self.kernel_size == 2:
            a = np.array([1.0, 1.0])
        elif self.kernel_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.kernel_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.kernel_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.kernel_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.kernel_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        self.kernel = tf.constant(a, dtype=self.compute_dtype)
        self.kernel = tf.divide(self.kernel, tf.reduce_sum(self.kernel))
        self.kernel = tf.repeat(tf.expand_dims(self.kernel, -1), self.filters, axis=-1)  # [kernel_size, filters]

    def call(self, inputs):
        inputs = tf.pad(inputs, paddings=self.paddings, mode=self.pad_mode)
        in_channels = tf.shape(inputs)[2]
        kernel = tf.repeat(tf.expand_dims(self.kernel, axis=1), in_channels, axis=1)  # [kernel_size, in_channel, filters]
        return tf.nn.conv1d(inputs, filters=kernel, stride=self.strides, padding="VALID")
