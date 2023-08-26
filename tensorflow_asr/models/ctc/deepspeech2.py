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

from tensorflow_asr.models.base_layer import Identity, Layer, Reshape
from tensorflow_asr.models.ctc.base_ctc import CtcModel
from tensorflow_asr.models.layers.convolution import DepthwiseConv1D
from tensorflow_asr.utils import layer_util, math_util

# ----------------------------------- CONV ----------------------------------- #


class RowConv1D(Layer):
    def __init__(
        self,
        future_width=2,
        activation="relu",
        regularizer=None,
        **kwargs,
    ):
        assert future_width >= 0, "Future context must be positive"
        super().__init__(**kwargs)
        self.conv = DepthwiseConv1D(
            kernel_size=future_width * 2 + 1,
            strides=1,
            padding="causal",
            use_bias=False,
            depthwise_regularizer=regularizer,
            bias_regularizer=regularizer,
            name="conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn", gamma_regularizer=regularizer, beta_regularizer=regularizer)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs, training=False):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = self.conv.compute_output_shape(input_shape)
        output_shape = self.bn.compute_output_shape(output_shape)
        return output_shape


class ConvBlock(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [11, 41],
        strides: list = [2, 2],
        filters: int = 32,
        padding: str = "causal",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = layer_util.get_conv(conv_type)(
            filters=filters,
            kernel_size=kernels,
            strides=strides,
            padding=padding,
            name=conv_type,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
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
            outputs_length, filter_size=self.conv.kernel_size[0], padding=self.conv.padding, stride=self.conv.strides[0]
        )
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (
            math_util.conv_output_length(length, filter_size=self.conv.kernel_size[0], padding=self.conv.padding, stride=self.conv.strides[0])
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
        kernel_regularizer=None,
        bias_regularizer=None,
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
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
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
                math_util.conv_output_length(length, filter_size=conv.conv.kernel_size[0], padding=conv.conv.padding, stride=conv.conv.strides[0])
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
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn = layer_util.get_rnn(rnn_type)(
            units,
            dropout=dropout,
            unroll=unroll,
            return_sequences=True,
            use_bias=True,
            name=rnn_type,
            zero_output_for_mask=True,
            dtype=tf.float32 if self.dtype == tf.bfloat16 else self.dtype,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, name=f"b{rnn_type}")
        self.rowconv = None
        if not bidirectional and rowconv > 0:
            self.rowconv = RowConv1D(future_width=rowconv, name="rowconv", regularizer=kernel_regularizer, activation="relu")

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        if self.dtype == tf.bfloat16:
            outputs = tf.cast(outputs, tf.float32)
        outputs = self.rnn(outputs, training=training)  # mask auto populate
        if self.dtype == tf.bfloat16:
            outputs = tf.cast(outputs, self.dtype)
        if self.rowconv is not None:
            outputs = self.rowconv(outputs, training=training)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.rnn.compute_output_shape(output_shape)
        return output_shape, output_length_shape


class RnnModule(Layer):
    def __init__(
        self,
        nlayers: int = 5,
        rnn_type: str = "lstm",
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [
            RnnBlock(
                rnn_type=rnn_type,
                units=units,
                bidirectional=bidirectional,
                unroll=unroll,
                rowconv=rowconv,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            for i in range(nlayers)
        ]

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return output_shape


# ------------------------------ FULLY CONNECTED ----------------------------- #


class FcBlock(Layer):
    def __init__(
        self,
        units: int = 1024,
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(units, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name="fc")
        self.relu = tf.keras.layers.ReLU(name="relu")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.fc(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.fc.compute_output_shape(output_shape)
        return output_shape, output_length_shape


class FcModule(Layer):
    def __init__(
        self,
        nlayers: int = 0,
        units: int = 1024,
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [
            FcBlock(units=units, dropout=dropout, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name=f"block_{i}")
            for i in range(nlayers)
        ]

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return output_shape


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
        rnn_units: int = 1024,
        rnn_bidirectional: bool = True,
        rnn_unroll: bool = False,
        rnn_rowconv: int = 0,
        rnn_dropout: float = 0.1,
        fc_nlayers: int = 0,
        fc_units: int = 1024,
        fc_dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
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
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="conv_module",
        )
        self.rnn_module = RnnModule(
            nlayers=rnn_nlayers,
            rnn_type=rnn_type,
            units=rnn_units,
            bidirectional=rnn_bidirectional,
            unroll=rnn_unroll,
            rowconv=rnn_rowconv,
            dropout=rnn_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="rnn_module",
        )
        self.fc_module = FcModule(
            nlayers=fc_nlayers,
            units=fc_units,
            dropout=fc_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
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
    def __init__(
        self,
        vocab_size: int,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab = tf.keras.layers.Dense(vocab_size, name="logits", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

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
