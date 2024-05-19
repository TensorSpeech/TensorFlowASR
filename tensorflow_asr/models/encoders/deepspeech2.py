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
from tensorflow_asr.models.layers.convolution import DepthwiseConv1D
from tensorflow_asr.utils import layer_util, math_util

# ----------------------------------- CONV ----------------------------------- #


class RowConv1D(Layer):
    def __init__(
        self,
        future_width=2,
        activation="relu",
        regularizer=None,
        initializer=None,
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
            depthwise_initializer=initializer,
            bias_regularizer=regularizer,
            bias_initializer=initializer,
            name="conv",
            dtype=self.dtype,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn",
            gamma_regularizer=regularizer,
            beta_regularizer=regularizer,
            dtype=self.dtype,
        )
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
        activation: str = "relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
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
            kernel_initializer=initializer,
            bias_regularizer=bias_regularizer,
            bias_initializer=initializer,
            dtype=self.dtype,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.act = tf.keras.layers.Activation(activation=activation, dtype=self.dtype)
        self.do = tf.keras.layers.Dropout(dropout, name="dropout", dtype=self.dtype)
        self.time_reduction_factor = self.conv.strides[0]

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act(outputs, training=training)
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
        output_shape = self.bn.compute_output_shape(output_shape)
        output_shape = self.act.compute_output_shape(output_shape)
        output_shape = self.do.compute_output_shape(output_shape)
        return output_shape, output_length_shape


class ConvModule(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [[11, 41], [11, 21], [11, 21]],
        strides: list = [[2, 2], [1, 2], [1, 2]],
        filters: list = [32, 32, 96],
        padding: str = "causal",
        activation: str = "relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert conv_type in ("conv1d", "conv2d")
        assert len(kernels) == len(strides) == len(filters)
        assert dropout >= 0.0

        self.pre = Reshape(name="preprocess", dtype=self.dtype) if conv_type == "conv1d" else Identity(name="iden", dtype=self.dtype)

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
                activation=activation,
                name=f"block_{i}",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                dtype=self.dtype,
            )
            self.convs.append(conv_block)
            self.time_reduction_factor *= conv_block.time_reduction_factor

        self.post = Reshape(name="postprocess", dtype=self.dtype) if conv_type == "conv2d" else Identity(name="iden", dtype=self.dtype)

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
        rowconv_activation: str = "relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn = layer_util.get_rnn(rnn_type)(
            units,
            dropout=dropout,
            unroll=unroll,
            return_sequences=True,
            return_state=True,
            use_bias=True,
            name=rnn_type,
            zero_output_for_mask=True,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=initializer,
            bias_regularizer=bias_regularizer,
            bias_initializer=initializer,
            dtype=self.dtype,
        )
        self._bidirectional = bidirectional
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, name=f"b{rnn_type}", dtype=self.dtype)
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.rowconv = None
        if not bidirectional and rowconv > 0:
            self.rowconv = RowConv1D(
                future_width=rowconv,
                name="rowconv",
                regularizer=kernel_regularizer,
                initializer=initializer,
                activation=rowconv_activation,
                dtype=self.dtype,
            )

    def get_initial_state(self, batch_size: int):
        if self._bidirectional:
            states = self.rnn.forward_layer.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype))
            states += self.rnn.backward_layer.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype))
        else:
            states = self.rnn.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype))
        return states

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs, *_ = self.rnn(outputs, training=training)  # mask auto populate
        outputs = self.bn(outputs, training=training)
        if self.rowconv is not None:
            outputs = self.rowconv(outputs, training=training)
        return outputs, outputs_length

    def call_next(self, inputs, previous_encoder_states):
        with tf.name_scope(f"{self.name}_call_next"):
            outputs, outputs_length = inputs
            outputs, *_states = self.rnn(outputs, training=False, initial_state=tf.unstack(previous_encoder_states, axis=0))
            outputs = self.bn(outputs, training=False)
            if self.rowconv is not None:
                outputs = self.rowconv(outputs, training=False)
            return outputs, outputs_length, tf.stack(_states)

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape, *_ = self.rnn.compute_output_shape(output_shape)
        output_shape = self.bn.compute_output_shape(output_shape)
        if self.rowconv is not None:
            output_shape = self.rowconv.compute_output_shape(output_shape)
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
        rowconv_activation: str = "relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
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
                rowconv_activation=rowconv_activation,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                name=f"block_{i}",
                dtype=self.dtype,
            )
            for i in range(nlayers)
        ]

    def get_initial_state(self, batch_size: int):
        """
        Get zeros states

        Returns
        -------
        tf.Tensor, shape [B, num_rnns, nstates, state_size]
            Zero initialized states
        """
        states = []
        for block in self.blocks:
            states.append(tf.stack(block.get_initial_state(batch_size=batch_size), axis=0))
        return tf.transpose(tf.stack(states, axis=0), perm=[2, 0, 1, 3])

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def call_next(self, inputs, previous_encoder_states):
        outputs = inputs
        previous_encoder_states = tf.transpose(previous_encoder_states, perm=[1, 2, 0, 3])
        new_states = []
        for i, block in enumerate(self.blocks):
            *outputs, _states = block.call_next(outputs, previous_encoder_states=previous_encoder_states[i])
            new_states.append(_states)
        return outputs, tf.transpose(tf.stack(new_states, axis=0), perm=[2, 0, 1, 3])

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
        activation: str = "relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(
            units,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=initializer,
            bias_regularizer=bias_regularizer,
            bias_initializer=initializer,
            name="fc",
            dtype=self.dtype,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.act = tf.keras.layers.Activation(activation=activation, dtype=self.dtype)
        self.do = tf.keras.layers.Dropout(dropout, name="dropout", dtype=self.dtype)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.fc(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act(outputs, training=training)
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
        activation: str = "relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [
            FcBlock(
                units=units,
                activation=activation,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                name=f"block_{i}",
                dtype=self.dtype,
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


class DeepSpeech2Encoder(Layer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
        conv_strides: list = [[2, 2], [1, 2], [1, 2]],
        conv_filters: list = [32, 32, 96],
        conv_padding: str = "same",
        conv_activation: str = "relu",
        conv_dropout: float = 0.1,
        conv_initializer: str = None,
        rnn_nlayers: int = 5,
        rnn_type: str = "lstm",
        rnn_units: int = 1024,
        rnn_bidirectional: bool = True,
        rnn_unroll: bool = False,
        rnn_rowconv: int = 0,
        rnn_rowconv_activation: str = "relu",
        rnn_dropout: float = 0.1,
        rnn_initializer: str = None,
        fc_nlayers: int = 0,
        fc_units: int = 1024,
        fc_activation: str = "relu",
        fc_dropout: float = 0.1,
        fc_initializer: str = None,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_module = ConvModule(
            conv_type=conv_type,
            kernels=conv_kernels,
            strides=conv_strides,
            filters=conv_filters,
            padding=conv_padding,
            activation=conv_activation,
            dropout=conv_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            initializer=conv_initializer or initializer,
            name="conv_module",
            dtype=self.dtype,
        )
        self.rnn_module = RnnModule(
            nlayers=rnn_nlayers,
            rnn_type=rnn_type,
            units=rnn_units,
            bidirectional=rnn_bidirectional,
            unroll=rnn_unroll,
            rowconv=rnn_rowconv,
            rowconv_activation=rnn_rowconv_activation,
            dropout=rnn_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            initializer=rnn_initializer or initializer,
            name="rnn_module",
            dtype=self.dtype,
        )
        self.fc_module = FcModule(
            nlayers=fc_nlayers,
            units=fc_units,
            activation=fc_activation,
            dropout=fc_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            initializer=fc_initializer or initializer,
            name="fc_module",
            dtype=self.dtype,
        )
        self.time_reduction_factor = self.conv_module.time_reduction_factor

    def get_initial_state(self, batch_size: int):
        """
        Get zeros states

        Returns
        -------
        tf.Tensor, shape [B, num_rnns, nstates, state_size]
            Zero initialized states
        """
        return self.rnn_module.get_initial_state(batch_size=batch_size)

    def call(self, inputs, training=False):
        *outputs, caching = inputs
        outputs = self.conv_module(outputs, training=training)
        outputs = self.rnn_module(outputs, training=training)
        outputs = self.fc_module(outputs, training=training)
        return *outputs, caching

    def call_next(self, features, features_length, previous_encoder_states, *args, **kwargs):
        """
        Recognize function for encoder network from previous encoder states

        Parameters
        ----------
        features : tf.Tensor, shape [B, T, F, C]
        features_length : tf.Tensor, shape [B]
        previous_encoder_states : tf.Tensor, shape [B, nlayers, nstates, rnn_units] -> [nlayers, nstates, B, rnn_units]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor], shape ([B, T, dmodel], [B], [nlayers, nstates, B, rnn_units] -> [B, nlayers, nstates, rnn_units])
        """
        with tf.name_scope(f"{self.name}_call_next"):
            outputs = (features, features_length)
            outputs = self.conv_module(outputs, training=False)
            outputs, new_encoder_states = self.rnn_module.call_next(outputs, previous_encoder_states=previous_encoder_states)
            outputs, outputs_length = self.fc_module(outputs, training=False)
            return outputs, outputs_length, new_encoder_states

    def compute_mask(self, inputs, mask=None):
        *outputs, caching = inputs
        return *self.conv_module.compute_mask(outputs, mask=mask), getattr(caching, "_keras_mask", None)

    def compute_output_shape(self, input_shape):
        *output_shape, caching_shape = input_shape
        output_shape = self.conv_module.compute_output_shape(output_shape)
        output_shape = self.rnn_module.compute_output_shape(output_shape)
        output_shape = self.fc_module.compute_output_shape(output_shape)
        return *output_shape, caching_shape
