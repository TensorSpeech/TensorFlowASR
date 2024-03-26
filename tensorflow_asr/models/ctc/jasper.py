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
from tensorflow_asr.models.ctc.base_ctc import CtcModel
from tensorflow_asr.models.encoders.jasper import JasperEncoder
from tensorflow_asr.models.layers.convolution import Conv1D


class JasperDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        padding: str = "causal",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab = Conv1D(
            filters=vocab_size,
            kernel_size=1,
            strides=1,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="logits",
            dtype=self.dtype,
        )
        self._vocab_size = vocab_size

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def call_next(self, logits, logits_length, *args, **kwargs):
        outputs, outputs_length = self((logits, logits_length), training=False)
        return outputs, outputs_length, None

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.models.ctc")
class Jasper(CtcModel):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        speech_config: dict,
        dense: bool = False,
        padding: str = "causal",
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
            blank=blank,
            speech_config=speech_config,
            encoder=JasperEncoder(
                dense=dense,
                padding=padding,
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
            decoder=JasperDecoder(
                vocab_size=vocab_size,
                padding=padding,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="decoder",
            ),
            name=name,
            **kwargs,
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor
