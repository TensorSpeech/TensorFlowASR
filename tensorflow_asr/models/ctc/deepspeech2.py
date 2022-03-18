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
from tensorflow_asr.models.layers.row_conv_1d import RowConv1D
from tensorflow_asr.models.layers.sequence_wise_bn import SequenceBatchNorm
from tensorflow_asr.utils import layer_util, math_util


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs):
        return math_util.merge_two_last_dims(inputs)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [11, 41],
        strides: list = [2, 2],
        filters: int = 32,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        CNN = layer_util.get_conv(conv_type)
        self.conv = CNN(filters=filters, kernel_size=kernels, strides=strides, padding="same", name=conv_type)
        self.bn = tf.keras.layers.BatchNormalization(name="bn")
        self.relu = tf.keras.layers.ReLU(name="relu")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")

    def call(self, inputs, training=False):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [[11, 41], [11, 21], [11, 21]],
        strides: list = [[2, 2], [1, 2], [1, 2]],
        filters: list = [32, 32, 96],
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(kernels) == len(strides) == len(filters)
        assert dropout >= 0.0

        self.preprocess = None  # reshape from [B, T, F, C] to [B, T, F * C]
        if conv_type == "conv1d":
            self.preprocess = Reshape(name="preprocess")

        self.blocks = [
            ConvBlock(conv_type=conv_type, kernels=kernels[i], strides=strides[i], filters=filters[i], dropout=dropout, name=f"block_{i}")
            for i in range(len(filters))
        ]

        self.postprocess = None  # reshape from [B, T, F, C] to [B, T, F * C]
        if conv_type == "conv2d":
            self.postprocess = Reshape(name="postprocess")

        self.reduction_factor = 1
        for s in strides:
            self.reduction_factor *= s[0]

    def call(self, inputs, training=False):
        outputs = inputs
        if self.preprocess is not None:
            outputs = self.preprocess(outputs)
        for block in self.blocks:
            outputs = block(outputs, training=training)
        if self.postprocess is not None:
            outputs = self.postprocess(outputs)
        return outputs


class RnnBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_type: str = "lstm",
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        RNN = layer_util.get_rnn(rnn_type)
        self.rnn = RNN(units, dropout=dropout, unroll=unroll, return_sequences=True, use_bias=True, name=rnn_type)
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, name=f"b{rnn_type}")
        self.bn = SequenceBatchNorm(time_major=False, name="bn")
        self.rowconv = None
        if not bidirectional and rowconv > 0:
            self.rowconv = RowConv1D(filters=units, future_context=rowconv, name="rowconv")

    def call(self, inputs, training=False):
        outputs = self.rnn(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        if self.rowconv is not None:
            outputs = self.rowconv(outputs, training=training)
        return outputs


class RnnModule(tf.keras.layers.Layer):
    def __init__(
        self,
        nlayers: int = 5,
        rnn_type: str = "lstm",
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [
            RnnBlock(rnn_type=rnn_type, units=units, bidirectional=bidirectional, unroll=unroll, rowconv=rowconv, dropout=dropout, name=f"block_{i}")
            for i in range(nlayers)
        ]

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs


class FcBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 1024,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(units, name="fc")
        self.bn = tf.keras.layers.BatchNormalization(name="bn")
        self.relu = tf.keras.layers.ReLU(name="relu")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")

    def call(self, inputs, training=False):
        outputs = self.fc(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs


class FcModule(tf.keras.layers.Layer):
    def __init__(
        self,
        nlayers: int = 0,
        units: int = 1024,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [FcBlock(units=units, dropout=dropout, name=f"block_{i}") for i in range(nlayers)]

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs


class DeepSpeech2Encoder(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
        conv_strides: list = [[2, 2], [1, 2], [1, 2]],
        conv_filters: list = [32, 32, 96],
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_module = ConvModule(
            conv_type=conv_type, kernels=conv_kernels, strides=conv_strides, filters=conv_filters, dropout=conv_dropout, name="conv_module"
        )
        self.rnn_module = RnnModule(
            nlayers=rnn_nlayers,
            rnn_type=rnn_type,
            units=rnn_units,
            bidirectional=rnn_bidirectional,
            unroll=rnn_unroll,
            rowconv=rnn_rowconv,
            dropout=rnn_dropout,
            name="rnn_module",
        )
        self._rnn_units = rnn_units
        self.fc_module = FcModule(nlayers=fc_nlayers, units=fc_units, dropout=fc_dropout, name="fc_module")
        self._fc_nlayers = fc_nlayers
        self._fc_units = fc_units
        self.time_reduction_factor = self.conv_module.reduction_factor

    def call(self, inputs, training=False):
        outputs, inputs_length = inputs
        outputs = self.conv_module(outputs, training=training)
        outputs = self.rnn_module(outputs, training=training)
        outputs = self.fc_module(outputs, training=training)
        return outputs, inputs_length

    def compute_output_shape(self, input_shape):
        inputs_shape, inputs_length_shape = input_shape
        outputs_time = None if inputs_shape[1] is None else math_util.legacy_get_reduced_length(inputs_shape[1], self.time_reduction_factor)
        outputs_batch = inputs_shape[0]
        outputs_size = self._fc_units if self._fc_nlayers > 0 else self._rnn_units
        outputs_shape = [outputs_batch, outputs_time, outputs_size]
        return tuple(outputs_shape), tuple(inputs_length_shape)


class DeepSpeech2Decoder(Layer):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab = tf.keras.layers.Dense(vocab_size, name="logits")
        self._vocab_size = vocab_size

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


class DeepSpeech2(CtcModel):
    def __init__(
        self,
        vocab_size: int,
        conv_type: str = "conv2d",
        conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
        conv_strides: list = [[2, 2], [1, 2], [1, 2]],
        conv_filters: list = [32, 32, 96],
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
        **kwargs,
    ):
        super().__init__(
            encoder=DeepSpeech2Encoder(
                conv_type=conv_type,
                conv_kernels=conv_kernels,
                conv_strides=conv_strides,
                conv_filters=conv_filters,
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
                name="encoder",
            ),
            decoder=DeepSpeech2Decoder(vocab_size=vocab_size, name="decoder"),
            name=name,
            **kwargs,
        )
        self.time_reduction_factor = self.encoder.conv_module.reduction_factor
