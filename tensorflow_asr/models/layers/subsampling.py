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

import tensorflow as tf

from tensorflow_asr.models.layers.base_layer import Layer
from tensorflow_asr.utils import math_util, shape_util


class Subsampling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_reduction_factor = 1

    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = self._create_mask(outputs, outputs_length)
        outputs, outputs_length = self._update_mask(outputs, outputs_length)
        return outputs, outputs_length

    def _create_mask(self, inputs, inputs_length):
        mask = getattr(inputs, "_keras_mask")
        if mask is None:
            mask = tf.sequence_mask(inputs_length, maxlen=tf.shape(inputs)[1], dtype=tf.bool)
        inputs._keras_mask = mask  # pylint: disable=protected-access
        return inputs

    def _update_mask(self, inputs, inputs_length):
        mask = getattr(inputs, "_keras_mask")
        if mask is None:
            raise ValueError("_keras_mask is required")
        mask = tf.slice(mask, begin=[0, 0], size=[-1, tf.shape(inputs)[1]])
        inputs_length = tf.reduce_sum(tf.cast(mask, inputs_length.dtype), axis=-1)
        inputs._keras_mask = mask  # pylint: disable=protected-access
        return inputs, inputs_length

    def compute_output_shape(self, input_shape):
        inputs_shape, inputs_length_shape = input_shape
        reduced_time = math_util.legacy_get_reduced_length(inputs_shape[1], self.time_reduction_factor)
        inputs_shape = list(inputs_shape)
        inputs_shape[1] = reduced_time
        return tuple(inputs_shape), inputs_length_shape


class TimeReduction(Subsampling):
    def __init__(self, factor: int, name: str = "TimeReduction", **kwargs):
        super().__init__(name=name, **kwargs)
        self.time_reduction_factor = factor

    def padding(self, time):
        new_time = tf.math.ceil(time / self.time_reduction_factor) * self.time_reduction_factor
        return tf.cast(new_time, dtype=tf.int32) - time

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
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.conv2 = tf.keras.layers.Conv2D(
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
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.conv4 = tf.keras.layers.Conv2D(
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
        self._do_apply_mask = False

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


class Conv2dSubsampling(Subsampling):
    def __init__(
        self,
        nlayers: int,
        filters: int,
        strides: list or tuple or int = 2,
        kernel_size: int or list or tuple = 3,
        padding: str = "same",
        norm: str = "batch",
        activation: str = "swish",
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
                tf.keras.layers.Conv2D(
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
            self.time_reduction_factor *= strides
        self._do_apply_mask = False

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        outputs = self._create_mask(inputs, inputs_length)
        for block in self.convs:
            outputs = block(outputs, training=training)
        outputs = math_util.merge_two_last_dims(outputs)
        outputs, outputs_length = super().call([outputs, inputs_length])
        return outputs, outputs_length


class Conv1dSubsampling(Subsampling):
    def __init__(
        self,
        nlayers: int,
        filters: int,
        strides: int = 2,
        kernel_size: int = 3,
        padding: str = "causal",
        norm: str = "batch",
        activation: str = "swish",
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
                tf.keras.layers.Conv1D(
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
                subblock.add(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
            elif norm == "layer":
                subblock.add(tf.keras.layers.LayerNormalization(name=f"ln_{i}"))
            subblock.add(tf.keras.layers.Activation(activation, name=f"{activation}_{i}"))
            self.convs.append(subblock)
            self.time_reduction_factor *= strides
        self._do_apply_mask = False

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        outputs = self._create_mask(inputs, inputs_length)
        outputs = math_util.merge_two_last_dims(outputs)
        for block in self.convs:
            outputs = block(outputs, training=training)
        outputs, outputs_length = super().call([outputs, inputs_length])
        return outputs, outputs_length
