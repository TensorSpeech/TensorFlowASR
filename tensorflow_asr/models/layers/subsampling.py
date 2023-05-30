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


class Subsampling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_reduction_factor = 1

    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = self._create_mask(outputs, outputs_length)
        outputs, outputs_length = self._update_mask_and_input_length(outputs, outputs_length)
        return outputs, outputs_length

    def _create_mask(self, inputs, inputs_length):
        mask = getattr(inputs, "_keras_mask", None)
        if mask is None:
            mask = tf.sequence_mask(inputs_length, maxlen=tf.shape(inputs)[1], dtype=tf.bool)
            inputs._keras_mask = mask  # pylint: disable=protected-access
        return inputs

    def _update_mask_and_input_length(self, inputs, inputs_length):
        raise NotImplementedError()

    def compute_output_shape(self, input_shape):
        inputs_shape, inputs_length_shape = input_shape
        reduced_time = math_util.legacy_get_reduced_length(inputs_shape[1], self.time_reduction_factor)
        inputs_shape = list(inputs_shape)
        inputs_shape[1] = reduced_time
        return inputs_shape, inputs_length_shape


class TimeReduction(Subsampling):
    def __init__(self, factor: int, name: str = "TimeReduction", **kwargs):
        super().__init__(name=name, **kwargs)
        self.time_reduction_factor = factor

    def padding(self, time):
        new_time = tf.math.ceil(time / self.time_reduction_factor) * self.time_reduction_factor
        return tf.cast(new_time, dtype=tf.int32) - time

    def _update_mask_and_input_length(self, inputs, inputs_length):
        outputs_length = math_util.get_reduced_length(inputs_length, self.time_reduction_factor)
        outputs = math_util.apply_mask(inputs, mask=tf.sequence_mask(outputs_length, maxlen=tf.shape(inputs)[1], dtype=tf.bool))
        return outputs, outputs_length

    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = self._create_mask(outputs, outputs_length)
        shape = shape_util.shape_list(outputs)
        outputs = tf.pad(outputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
        outputs = tf.reshape(outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor])
        outputs, outputs_length = super().call([outputs, outputs_length])
        return outputs, outputs_length


class VggSubsampling(Subsampling):
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
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, name="maxpool_1")
        self.conv3 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
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
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, name="maxpool_2")
        self.time_reduction_factor = self.maxpool1.pool_size[0] * self.maxpool2.pool_size[0]

    def _update_mask_and_input_length(self, inputs, inputs_length):
        outputs_length = math_util.conv_output_length(
            inputs_length,
            self.maxpool1.pool_size[0],
            padding=self.maxpool1.padding,
            stride=self.maxpool1.strides[0],
        )
        outputs_length = math_util.conv_output_length(
            outputs_length,
            self.maxpool2.pool_size[0],
            padding=self.maxpool2.padding,
            stride=self.maxpool2.strides[0],
        )
        outputs = math_util.apply_mask(inputs, mask=tf.sequence_mask(outputs_length, maxlen=tf.shape(inputs)[1], dtype=tf.bool))
        return outputs, outputs_length

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        outputs = self._create_mask(inputs, inputs_length)

        outputs = self.conv1(outputs, training=training)
        outputs = self.conv2(outputs, training=training)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv3(outputs, training=training)
        outputs = self.conv4(outputs, training=training)
        outputs = self.maxpool2(outputs, training=training)

        outputs = math_util.merge_two_last_dims(outputs)
        outputs, outputs_length = super().call([outputs, inputs_length])
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        inputs_shape, inputs_length_shape = input_shape
        outputs_shape = self.conv1.compute_output_shape(inputs_shape)
        outputs_shape = self.conv2.compute_output_shape(outputs_shape)
        outputs_shape = self.maxpool1.compute_output_shape(outputs_shape)
        outputs_shape = self.conv3.compute_output_shape(outputs_shape)
        outputs_shape = self.conv4.compute_output_shape(outputs_shape)
        outputs_shape = self.maxpool2.compute_output_shape(outputs_shape)
        outputs_shape = list(outputs_shape[:2]) + [outputs_shape[2] * outputs_shape[3]]
        return outputs_shape, inputs_length_shape


class Conv2dSubsampling(Subsampling):
    def __init__(
        self,
        nlayers: int,
        filters: int,
        strides: list or tuple or int = 2,
        kernel_size: int or list or tuple = 3,
        padding: str = "same",
        norm: str = "none",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conv2d_subsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.convs = []
        self.time_reduction_factor = 1
        for i in range(nlayers):
            subblock = tf.keras.Sequential(name=f"block_{i}")
            subblock.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    name=f"conv_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                )
            )
            if norm == "batch":
                subblock.add(
                    tf.keras.layers.BatchNormalization(
                        name=f"bn_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                    )
                )
            elif norm == "layer":
                subblock.add(
                    tf.keras.layers.LayerNormalization(
                        name=f"ln_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                    )
                )
            subblock.add(tf.keras.layers.Activation(activation, name=f"{activation}_{i}"))
            self.convs.append(subblock)
            self.time_reduction_factor *= subblock.layers[0].strides[0]

    def _update_mask_and_input_length(self, inputs, inputs_length):
        outputs_length = inputs_length
        for block in self.convs:
            outputs_length = math_util.conv_output_length(
                outputs_length,
                filter_size=block.layers[0].kernel_size[0],
                padding=block.layers[0].padding,
                stride=block.layers[0].strides[0],
            )
        outputs = math_util.apply_mask(inputs, mask=tf.sequence_mask(outputs_length, maxlen=tf.shape(inputs)[1], dtype=tf.bool))
        return outputs, outputs_length

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        outputs = self._create_mask(inputs, inputs_length)
        for block in self.convs:
            outputs = block(outputs, training=training)
        outputs = math_util.merge_two_last_dims(outputs)
        outputs, outputs_length = super().call([outputs, inputs_length])
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        outputs_shape, inputs_length_shape = input_shape
        for block in self.convs:
            outputs_shape = block.layers[0].compute_output_shape(outputs_shape)
        outputs_shape = list(outputs_shape[:2]) + [outputs_shape[2] * outputs_shape[3]]
        return tuple(outputs_shape), inputs_length_shape


class Conv1dSubsampling(Subsampling):
    def __init__(
        self,
        nlayers: int,
        filters: int,
        strides: int = 2,
        kernel_size: int = 3,
        padding: str = "causal",
        norm: str = "none",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conv1d_subsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.convs = []
        self.time_reduction_factor = 1
        for i in range(nlayers):
            subblock = tf.keras.Sequential(name=f"block_{i}")
            subblock.add(
                Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    name=f"conv_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                )
            )
            if norm == "batch":
                subblock.add(
                    tf.keras.layers.BatchNormalization(
                        name=f"bn_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                    )
                )
            elif norm == "layer":
                subblock.add(
                    tf.keras.layers.LayerNormalization(
                        name=f"ln_{i}",
                        gamma_regularizer=kernel_regularizer,
                        beta_regularizer=bias_regularizer,
                    )
                )
            subblock.add(tf.keras.layers.Activation(activation, name=f"{activation}_{i}"))
            self.convs.append(subblock)
            self.time_reduction_factor *= subblock.layers[0].strides[0]

    def _update_mask_and_input_length(self, inputs, inputs_length):
        outputs_length = inputs_length
        for block in self.convs:
            outputs_length = math_util.conv_output_length(
                outputs_length,
                filter_size=block.layers[0].kernel_size[0],
                padding=block.layers[0].padding,
                stride=block.layers[0].strides[0],
            )
        outputs = math_util.apply_mask(inputs, mask=tf.sequence_mask(outputs_length, maxlen=tf.shape(inputs)[1], dtype=tf.bool))
        return outputs, outputs_length

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        outputs = self._create_mask(inputs, inputs_length)
        outputs = math_util.merge_two_last_dims(outputs)
        for block in self.convs:
            outputs = block(outputs, training=training)
        outputs, outputs_length = super().call([outputs, inputs_length])
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        outputs_shape, inputs_length_shape = input_shape
        outputs_shape = list(outputs_shape[:2]) + [outputs_shape[2] * outputs_shape[3]]
        for block in self.convs:
            outputs_shape = block.layers[0].compute_output_shape(outputs_shape)
        return tuple(outputs_shape), inputs_length_shape
