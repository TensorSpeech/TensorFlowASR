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

import tensorflow as tf

from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.layers.convolution import Conv1D, Conv2D
from tensorflow_asr.utils import math_util, shape_util


class TimeReduction(Layer):
    def __init__(self, factor: int, name: str = "TimeReduction", **kwargs):
        super().__init__(name=name, **kwargs)
        self.time_reduction_factor = factor

    def padding(self, time):
        new_time = tf.math.ceil(time / self.time_reduction_factor) * self.time_reduction_factor
        return tf.cast(new_time, dtype=tf.int32) - time

    def call(self, inputs):
        outputs, outputs_length = inputs
        shape = shape_util.shape_list(outputs)
        outputs = tf.pad(outputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
        outputs = tf.reshape(outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor])
        outputs_length = math_util.get_reduced_length(outputs_length, reduction_factor=self.time_reduction_factor)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (math_util.get_reduced_length(length, self.time_reduction_factor) for length in (maxlen, outputs_length))
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        reduced_time = math_util.legacy_get_reduced_length(output_shape[1], self.time_reduction_factor)
        output_shape = output_shape[:1] + (reduced_time,) + output_shape[2:]
        return output_shape, output_length_shape


class VggSubsampling(Layer):
    def __init__(
        self,
        filters: tuple or list = (32, 64),
        kernel_size: int or list or tuple = 3,
        pool_size: int or list or tuple = 2,
        strides: int or list or tuple = 2,
        padding: str = "same",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="VggSubsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.conv1 = Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            dtype=self.dtype,
        )
        self.conv2 = Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            dtype=self.dtype,
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, dtype=self.dtype, name="maxpool_1")
        self.conv3 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            dtype=self.dtype,
        )
        self.conv4 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_4",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            dtype=self.dtype,
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, dtype=self.dtype, name="maxpool_2")
        self.time_reduction_factor = self.maxpool1.pool_size[0] * self.maxpool2.pool_size[0]

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs

        outputs = self.conv1(outputs, training=training)
        outputs = self.conv2(outputs, training=training)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv3(outputs, training=training)
        outputs = self.conv4(outputs, training=training)
        outputs = self.maxpool2(outputs, training=training)

        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        for pool in (self.maxpool1, self.maxpool2):
            maxlen, outputs_length = (
                math_util.conv_output_length(length, pool.pool_size[0], padding=pool.padding, stride=pool.strides[0])
                for length in (maxlen, outputs_length)
            )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        outputs_shape = self.conv1.compute_output_shape(output_shape)
        outputs_shape = self.conv2.compute_output_shape(outputs_shape)
        outputs_shape = self.maxpool1.compute_output_shape(outputs_shape)
        outputs_shape = self.conv3.compute_output_shape(outputs_shape)
        outputs_shape = self.conv4.compute_output_shape(outputs_shape)
        outputs_shape = self.maxpool2.compute_output_shape(outputs_shape)
        outputs_shape = outputs_shape[:2] + (outputs_shape[2] * outputs_shape[3],)
        return outputs_shape, output_length_shape


class Conv2dSubsampling(Layer):
    def __init__(
        self,
        filters: list,
        strides: list = [[2, 1], [2, 1]],
        kernels: list = [[3, 3], [3, 3]],
        paddings: list = ["causal", "causal"],
        norms: list = ["none", "none"],
        activations: list = ["relu", "relu"],
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conv2d_subsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert len(filters) == len(strides) == len(kernels) == len(paddings) == len(norms) == len(activations)
        self.convs = []
        self.time_reduction_factor = 1
        for i in range(len(filters)):
            subblock = tf.keras.Sequential(name=f"block_{i}")
            subblock.add(
                Conv2D(
                    filters=filters[i],
                    kernel_size=kernels[i],
                    strides=strides[i],
                    padding=paddings[i],
                    name=f"conv_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    dtype=self.dtype,
                )
            )
            if norms[i] == "batch":
                subblock.add(
                    tf.keras.layers.BatchNormalization(
                        name=f"bn_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                        dtype=self.dtype,
                    )
                )
            elif norms[i] == "layer":
                subblock.add(
                    tf.keras.layers.LayerNormalization(
                        name=f"ln_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                        dtype=self.dtype,
                    )
                )
            subblock.add(tf.keras.layers.Activation(activations[i], name=f"{activations[i]}_{i}", dtype=self.dtype))
            self.convs.append(subblock)
            self.time_reduction_factor *= subblock.layers[0].strides[0]

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        for block in self.convs:
            outputs = block(outputs, training=training)
            outputs_length = math_util.conv_output_length(
                outputs_length,
                filter_size=block.layers[0].kernel_size[0],
                padding=block.layers[0].padding,
                stride=block.layers[0].strides[0],
            )
        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        for block in self.convs:
            maxlen, outputs_length = (
                math_util.conv_output_length(
                    length, filter_size=block.layers[0].kernel_size[0], padding=block.layers[0].padding, stride=block.layers[0].strides[0]
                )
                for length in (maxlen, outputs_length)
            )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        for block in self.convs:
            output_shape = block.layers[0].compute_output_shape(output_shape)
        output_shape = output_shape[:2] + (output_shape[2] * output_shape[3],)
        return output_shape, output_length_shape


class Conv1dSubsampling(Layer):
    def __init__(
        self,
        filters: list,
        strides: list = [2, 2],
        kernels: list = [3, 3],
        paddings: list = ["causal", "causal"],
        norms: list = ["none", "none"],
        activations: list = ["relu", "relu"],
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conv1d_subsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert len(filters) == len(strides) == len(kernels) == len(paddings) == len(norms) == len(activations)
        self.convs = []
        self.time_reduction_factor = 1
        for i in range(len(filters)):
            subblock = tf.keras.Sequential(name=f"block_{i}")
            subblock.add(
                Conv1D(
                    filters=filters[i],
                    kernel_size=kernels[i],
                    strides=strides[i],
                    padding=paddings[i],
                    name=f"conv_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    dtype=self.dtype,
                )
            )
            if norms[i] == "batch":
                subblock.add(
                    tf.keras.layers.BatchNormalization(
                        name=f"bn_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                        dtype=self.dtype,
                    )
                )
            elif norms[i] == "layer":
                subblock.add(
                    tf.keras.layers.LayerNormalization(
                        name=f"ln_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                        dtype=self.dtype,
                    )
                )
            subblock.add(tf.keras.layers.Activation(activations[i], name=f"{activations[i]}_{i}", dtype=self.dtype))
            self.convs.append(subblock)
            self.time_reduction_factor *= subblock.layers[0].strides[0]

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = math_util.merge_two_last_dims(outputs)
        for block in self.convs:
            outputs = block(outputs, training=training)
            outputs_length = math_util.conv_output_length(
                outputs_length,
                filter_size=block.layers[0].kernel_size[0],
                padding=block.layers[0].padding,
                stride=block.layers[0].strides[0],
            )
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        for block in self.convs:
            maxlen, outputs_length = (
                math_util.conv_output_length(
                    length, filter_size=block.layers[0].kernel_size[0], padding=block.layers[0].padding, stride=block.layers[0].strides[0]
                )
                for length in (maxlen, outputs_length)
            )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = output_shape[:2] + (output_shape[2] * output_shape[3],)
        for block in self.convs:
            output_shape = block.layers[0].compute_output_shape(output_shape)
        return output_shape, output_length_shape
