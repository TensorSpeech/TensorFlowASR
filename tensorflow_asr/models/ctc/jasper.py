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

from tensorflow_asr.models.ctc.base_ctc import CtcModel
from tensorflow_asr.models.layers.base_layer import Layer
from tensorflow_asr.utils import math_util


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs):
        return math_util.merge_two_last_dims(inputs)


class JasperSubBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int = 256,
        kernels: int = 11,
        strides: int = 1,
        dropout: float = 0.1,
        dilation: int = 1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=kernels,
            strides=strides,
            dilation_rate=dilation,
            padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="conv1d",
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn")
        self.relu = tf.keras.layers.ReLU(name="relu")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.reduction_factor = strides

    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.conv1d(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs


class JasperResidual(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int = 256,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pointwise_conv1d = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="pointwise_conv1d",
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn")

    def call(self, inputs, training=False):
        outputs = self.pointwise_conv1d(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        return outputs


class JasperSubBlockResidual(JasperSubBlock):
    def __init__(
        self,
        channels: int = 256,
        kernels: int = 11,
        strides: int = 1,
        dropout: float = 0.1,
        dilation: int = 1,
        nresiduals: int = 1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(
            channels=channels,
            kernels=kernels,
            strides=strides,
            dropout=dropout,
            dilation=dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            **kwargs,
        )

        self.residuals = [
            JasperResidual(
                channels=channels,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"residual_{i}",
            )
            for i in range(nresiduals)
        ]

        self.add = tf.keras.layers.Add(name="add")

    def call(self, inputs, training=False):
        outputs, residuals = inputs
        outputs = self.conv1d(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        for i, res in enumerate(residuals):
            res = self.residuals[i](res, training=training)
            outputs = self.add([outputs, res], training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs


class JasperBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nsubblocks: int = 3,
        channels: int = 256,
        kernels: int = 11,
        dropout: float = 0.1,
        dense: bool = False,
        nresiduals: int = 1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dense = dense

        self.subblocks = [
            JasperSubBlock(
                channels=channels,
                kernels=kernels,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"subordinate_{i}",
            )
            for i in range(nsubblocks - 1)
        ]

        self.subblock_residual = JasperSubBlockResidual(
            channels=channels,
            kernels=kernels,
            dropout=dropout,
            nresiduals=nresiduals,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"subordinate_{nsubblocks - 1}",
        )

        self.reduction_factor = 1

    def call(self, inputs, training=False):
        inputs, residuals = inputs
        outputs = inputs
        for subblock in self.subblocks:
            outputs = subblock(outputs, training=training)
        if self.dense:
            residuals.append(inputs)
            outputs = self.subblock_residual([outputs, residuals], training=training)
        else:
            outputs = self.subblock_residual([outputs, [inputs]], training=training)
        return outputs, residuals


class JasperEncoder(Layer):
    def __init__(
        self,
        dense: bool = False,
        first_additional_block_channels: int = 256,
        first_additional_block_kernels: int = 11,
        first_additional_block_strides: int = 2,
        first_additional_block_dilation: int = 1,
        first_additional_block_dropout: int = 0.2,
        nsubblocks: int = 5,
        block_channels: list = [256, 384, 512, 640, 768],
        block_kernels: list = [11, 13, 17, 21, 25],
        block_dropout: list = [0.2, 0.2, 0.2, 0.3, 0.3],
        second_additional_block_channels: int = 896,
        second_additional_block_kernels: int = 1,
        second_additional_block_strides: int = 1,
        second_additional_block_dilation: int = 2,
        second_additional_block_dropout: int = 0.4,
        third_additional_block_channels: int = 1024,
        third_additional_block_kernels: int = 1,
        third_additional_block_strides: int = 1,
        third_additional_block_dilation: int = 1,
        third_additional_block_dropout: int = 0.4,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(block_channels) == len(block_kernels) == len(block_dropout)

        self.reshape = Reshape(name="reshape")

        self.first_additional_block = JasperSubBlock(
            channels=first_additional_block_channels,
            kernels=first_additional_block_kernels,
            strides=first_additional_block_strides,
            dropout=first_additional_block_dropout,
            dilation=first_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="first_block",
        )

        self.blocks = [
            JasperBlock(
                nsubblocks=nsubblocks,
                channels=block_channels[i],
                kernels=block_kernels[i],
                dropout=block_dropout[i],
                dense=dense,
                nresiduals=(i + 1) if dense else 1,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            for i in range(len(block_channels))
        ]

        self.second_additional_block = JasperSubBlock(
            channels=second_additional_block_channels,
            kernels=second_additional_block_kernels,
            strides=second_additional_block_strides,
            dropout=second_additional_block_dropout,
            dilation=second_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="second_block",
        )

        self.third_additional_block = JasperSubBlock(
            channels=third_additional_block_channels,
            kernels=third_additional_block_kernels,
            strides=third_additional_block_strides,
            dropout=third_additional_block_dropout,
            dilation=third_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="third_block",
        )
        self.time_reduction_factor = self.first_additional_block.reduction_factor
        self.time_reduction_factor *= self.second_additional_block.reduction_factor
        self.time_reduction_factor *= self.third_additional_block.reduction_factor

    def call(self, inputs, training=False):
        outputs, inputs_length = inputs
        outputs = self.reshape(outputs)
        outputs = self.first_additional_block(outputs, training=training)

        residuals = []
        for block in self.blocks:
            outputs, residuals = block([outputs, residuals], training=training)

        outputs = self.second_additional_block(outputs, training=training)
        outputs = self.third_additional_block(outputs, training=training)
        return outputs, inputs_length

    def compute_output_shape(self, input_shape):
        inputs_shape, inputs_length_shape = input_shape
        outputs_time = None if inputs_shape[1] is None else math_util.legacy_get_reduced_length(inputs_shape[1], self.time_reduction_factor)
        outputs_batch = inputs_shape[0]
        outputs_size = self.third_additional_block.conv1d.filters
        outputs_shape = [outputs_batch, outputs_time, outputs_size]
        return tuple(outputs_shape), tuple(inputs_length_shape)


class JasperDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab = tf.keras.layers.Conv1D(
            filters=vocab_size,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="logits",
        )
        self._vocab_size = vocab_size

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


class Jasper(CtcModel):
    def __init__(
        self,
        vocab_size: int,
        dense: bool = False,
        first_additional_block_channels: int = 256,
        first_additional_block_kernels: int = 11,
        first_additional_block_strides: int = 2,
        first_additional_block_dilation: int = 1,
        first_additional_block_dropout: int = 0.2,
        nsubblocks: int = 5,
        block_channels: list = [256, 384, 512, 640, 768],
        block_kernels: list = [11, 13, 17, 21, 25],
        block_dropout: list = [0.2, 0.2, 0.2, 0.3, 0.3],
        second_additional_block_channels: int = 896,
        second_additional_block_kernels: int = 1,
        second_additional_block_strides: int = 1,
        second_additional_block_dilation: int = 2,
        second_additional_block_dropout: int = 0.4,
        third_additional_block_channels: int = 1024,
        third_additional_block_kernels: int = 1,
        third_additional_block_strides: int = 1,
        third_additional_block_dilation: int = 1,
        third_additional_block_dropout: int = 0.4,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="jasper",
        **kwargs,
    ):
        super().__init__(
            encoder=JasperEncoder(
                dense=dense,
                first_additional_block_channels=first_additional_block_channels,
                first_additional_block_kernels=first_additional_block_kernels,
                first_additional_block_strides=first_additional_block_strides,
                first_additional_block_dilation=first_additional_block_dilation,
                first_additional_block_dropout=first_additional_block_dropout,
                nsubblocks=nsubblocks,
                block_channels=block_channels,
                block_kernels=block_kernels,
                block_dropout=block_dropout,
                second_additional_block_channels=second_additional_block_channels,
                second_additional_block_kernels=second_additional_block_kernels,
                second_additional_block_strides=second_additional_block_strides,
                second_additional_block_dilation=second_additional_block_dilation,
                second_additional_block_dropout=second_additional_block_dropout,
                third_additional_block_channels=third_additional_block_channels,
                third_additional_block_kernels=third_additional_block_kernels,
                third_additional_block_strides=third_additional_block_strides,
                third_additional_block_dilation=third_additional_block_dilation,
                third_additional_block_dropout=third_additional_block_dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="encoder",
            ),
            decoder=JasperDecoder(vocab_size=vocab_size, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name="decoder"),
            name=name,
            **kwargs,
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor
