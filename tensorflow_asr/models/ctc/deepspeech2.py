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
from tensorflow_asr.models.encoders.deepspeech2 import DeepSpeech2Encoder


class DeepSpeech2Decoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab = tf.keras.layers.Dense(
            vocab_size,
            name="logits",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.vocab.compute_output_shape(output_shape)
        return output_shape, output_length_shape


@tf.keras.utils.register_keras_serializable("tensorflow_asr.models.ctc")
class DeepSpeech2(CtcModel):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        speech_config: dict,
        conv_type: str = "conv2d",
        conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
        conv_strides: list = [[3, 2], [1, 2], [1, 2]],
        conv_filters: list = [32, 32, 96],
        conv_padding: str = "same",
        conv_dropout: float = 0.1,
        rnn_nlayers: int = 5,
        rnn_type: str = "lstm",
        rnn_units: int = 1024,
        rnn_bidirectional: bool = True,
        rnn_unroll: bool = False,
        rnn_rowconv: int = 0,
        rnn_dropout: float = 0.1,
        fc_nlayers: int = 0,
        fc_units: int = 1024,
        fc_dropout: float = 0.1,
        name: str = "deepspeech2",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(
            blank=blank,
            speech_config=speech_config,
            encoder=DeepSpeech2Encoder(
                conv_type=conv_type,
                conv_kernels=conv_kernels,
                conv_strides=conv_strides,
                conv_filters=conv_filters,
                conv_padding=conv_padding,
                conv_dropout=conv_dropout,
                rnn_nlayers=rnn_nlayers,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                rnn_bidirectional=rnn_bidirectional,
                rnn_unroll=rnn_unroll,
                rnn_rowconv=rnn_rowconv,
                rnn_dropout=rnn_dropout,
                fc_nlayers=fc_nlayers,
                fc_units=fc_units,
                fc_dropout=fc_dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="encoder",
            ),
            decoder=DeepSpeech2Decoder(
                vocab_size=vocab_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="decoder",
            ),
            name=name,
            **kwargs,
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor
