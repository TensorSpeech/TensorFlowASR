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

from tensorflow_asr.models.base_layer import Identity, Layer
from tensorflow_asr.models.ctc.base_ctc import CtcModel
from tensorflow_asr.models.layers.row_conv_1d import RowConv1D
from tensorflow_asr.models.layers.sequence_wise_bn import SequenceBatchNorm
from tensorflow_asr.utils import layer_util, math_util


class Reshape(Layer):
    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        b, t, f, c = output_shape
        return (b, t, f * c), output_length_shape


# ----------------------------------- CONV ----------------------------------- #


class ConvBlock(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [11, 41],
        strides: list = [2, 2],
        filters: int = 32,
        padding: str = "causal",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        CnnClass = layer_util.get_conv(conv_type)
        self.conv = CnnClass(filters=filters, kernel_size=kernels, strides=strides, padding=padding, name=conv_type)
        self.bn = tf.keras.layers.BatchNormalization(name="bn")
        self.relu = tf.keras.layers.ReLU(name="relu")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.time_reduction_factor = self.conv.strides[0]

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs_length = math_util.conv_output_length(
            outputs_length, filter_size=self.conv.filters, padding=self.conv.padding, stride=self.conv.strides[0]
        )
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (
            math_util.conv_output_length(length, filter_size=self.conv.filters, padding=self.conv.padding, stride=self.conv.strides[0])
            for length in (maxlen, outputs_length)
        )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.conv.compute_output_shape(output_shape)
        return output_shape, output_length_shape


class ConvModule(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [[11, 41], [11, 21], [11, 21]],
        strides: list = [[2, 2], [1, 2], [1, 2]],
        filters: list = [32, 32, 96],
        padding: str = "causal",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert conv_type in ("conv1d", "conv2d")
        assert len(kernels) == len(strides) == len(filters)
        assert dropout >= 0.0

        self.pre = Reshape(name="preprocess") if conv_type == "conv1d" else Identity(name="iden")

        self.convs = []
        self.time_reduction_factor = 1
        for i in range(len(filters)):
            conv_block = ConvBlock(
                conv_type=conv_type,
                kernels=kernels[i],
                strides=strides[i],
                filters=filters[i],
                dropout=dropout,
                padding=padding,
                name=f"block_{i}",
            )
            self.convs.append(conv_block)
            self.time_reduction_factor *= conv_block.time_reduction_factor

        self.post = Reshape(name="postprocess") if conv_type == "conv2d" else Identity(name="iden")

    def call(self, inputs, training=False):
        outputs = self.pre(inputs, training=training)
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.post(outputs, training=training)
        return outputs

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        for conv in self.convs:
            maxlen, outputs_length = (
                math_util.conv_output_length(length, filter_size=conv.conv.filters, padding=conv.conv.padding, stride=conv.conv.strides[0])
                for length in (maxlen, outputs_length)
            )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.pre.compute_output_shape(output_shape)
        for conv in self.convs:
            output_shape = conv.compute_output_shape(output_shape)
        output_shape = self.post.compute_output_shape(output_shape)
        return output_shape


# ------------------------------------ RNN ----------------------------------- #


class RnnBlock(Layer):
    def __init__(
        self,
        rnn_type: str = "lstm",
        bn_type: str = "bn",
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        RnnClass = layer_util.get_rnn(rnn_type)
        self.rnn = RnnClass(
            units,
            dropout=dropout,
            unroll=unroll,
            return_sequences=True,
            use_bias=True,
            name=rnn_type,
            zero_output_for_mask=True,
            dtype=self.dtype,
        )
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, name=f"b{rnn_type}")
        if bn_type not in ("bn", "sbn"):
            raise ValueError(f"bn_type must be in {('bn', 'sbn')}")
        self.bn = SequenceBatchNorm(time_major=False, name="bn") if bn_type == "sbn" else tf.keras.layers.BatchNormalization(name="bn")
        self.rowconv = None
        if not bidirectional and rowconv > 0:
            self.rowconv = RowConv1D(filters=units, future_context=rowconv, name="rowconv")

    def call(self, inputs, training=False, mask=None):
        outputs, outputs_length = inputs
        outputs = self.rnn(outputs, training=training, mask=mask[0] if mask else getattr(outputs, "_keras_mask", None))
        outputs = self.bn(outputs, training=training)
        if self.rowconv is not None:
            outputs = self.rowconv(outputs, training=training)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        return input_shape


class RnnModule(Layer):
    def __init__(
        self,
        nlayers: int = 5,
        rnn_type: str = "lstm",
        bn_type: str = "bn",
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [
            RnnBlock(
                rnn_type=rnn_type,
                bn_type=bn_type,
                units=units,
                bidirectional=bidirectional,
                unroll=unroll,
                rowconv=rowconv,
                dropout=dropout,
                name=f"block_{i}",
            )
            for i in range(nlayers)
        ]

    def call(self, inputs, training=False, mask=None):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training, mask=mask)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


# ------------------------------ FULLY CONNECTED ----------------------------- #


class FcBlock(Layer):
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
        outputs, outputs_length = inputs
        outputs = self.fc(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        return input_shape


class FcModule(Layer):
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

    def compute_output_shape(self, input_shape):
        return input_shape


class DeepSpeech2Encoder(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
        conv_strides: list = [[2, 2], [1, 2], [1, 2]],
        conv_filters: list = [32, 32, 96],
        conv_padding: str = "same",
        conv_dropout: float = 0.1,
        rnn_nlayers: int = 5,
        rnn_type: str = "lstm",
        rnn_bn_type: str = "sbn",
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
            conv_type=conv_type,
            kernels=conv_kernels,
            strides=conv_strides,
            filters=conv_filters,
            padding=conv_padding,
            dropout=conv_dropout,
            name="conv_module",
        )
        self.rnn_module = RnnModule(
            nlayers=rnn_nlayers,
            rnn_type=rnn_type,
            bn_type=rnn_bn_type,
            units=rnn_units,
            bidirectional=rnn_bidirectional,
            unroll=rnn_unroll,
            rowconv=rnn_rowconv,
            dropout=rnn_dropout,
            name="rnn_module",
        )
        self.fc_module = FcModule(
            nlayers=fc_nlayers,
            units=fc_units,
            dropout=fc_dropout,
            name="fc_module",
        )
        self.time_reduction_factor = self.conv_module.time_reduction_factor

    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.conv_module(outputs, training=training)
        outputs = self.rnn_module(outputs, training=training)
        outputs = self.fc_module(outputs, training=training)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self.conv_module.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.conv_module.compute_output_shape(output_shape)
        output_shape = self.rnn_module.compute_output_shape(output_shape)
        output_shape = self.fc_module.compute_output_shape(output_shape)
        return output_shape


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
        rnn_bn_type: str = "bn",
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
                rnn_bn_type=rnn_bn_type,
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
        self.time_reduction_factor = self.encoder.time_reduction_factor
