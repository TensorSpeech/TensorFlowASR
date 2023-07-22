# pylint: disable=attribute-defined-outside-init
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

from tensorflow_asr.models.activations.glu import GLU
from tensorflow_asr.models.base_layer import Identity, Layer

# from tensorflow_asr.models.base_model import BaseModelLayer as Layer
from tensorflow_asr.models.layers.convolution import Conv1D, DepthwiseConv1D
from tensorflow_asr.models.layers.multihead_attention import MultiHeadAttention, MultiHeadRelativeAttention
from tensorflow_asr.models.layers.positional_encoding import PositionalEncoding, RelativePositionalEncoding
from tensorflow_asr.models.layers.residual import Residual
from tensorflow_asr.models.layers.subsampling import Conv1dSubsampling, Conv2dSubsampling, VggSubsampling

L2 = tf.keras.regularizers.l2(1e-6)


class FFModule(Layer):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   fflayer(.)              # 4 * input_dim
      |   swish(.)
      |   dropout(.)
      |   fflayer(.)              # input_dim
      |   dropout(.)
      |   * 1/2
      \   /
        +
        |
      output
    """

    def __init__(
        self,
        input_dim,
        dropout=0.0,
        scale_factor=4,
        residual_factor=0.5,
        norm_position="pre",
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="ff_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        self.pre_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if norm_position == "pre"
            else Identity(name="preiden" if norm_position == "none" else "iden")
        )
        self.ffn1 = tf.keras.layers.Dense(
            units=scale_factor * input_dim,
            name="dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation="swish",
        )
        self.do1 = tf.keras.layers.Dropout(rate=dropout, name="dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            units=input_dim,
            name="dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do2 = tf.keras.layers.Dropout(rate=dropout, name="dropout_2")
        self.post_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if norm_position == "post"
            else Identity(name="postiden" if norm_position == "none" else "iden")
        )
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual")

    def call(self, inputs, training=False):
        outputs = self.pre_norm(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        outputs = self.residual([inputs, outputs], training=training)
        return outputs


class MHSAModule(Layer):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   mhsa(.)                 # head_size = dmodel // num_heads
      |   dropout(.)
      \   /
        +
        |
      output
    """

    def __init__(
        self,
        dmodel,
        head_size,
        num_heads,
        residual_factor=1.0,
        dropout=0.0,
        mha_type="relmha",
        norm_position="pre",
        memory_length=None,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="mhsa_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        assert mha_type in ("relmha", "mha")
        self.pre_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if norm_position == "pre"
            else Identity(name="preiden" if norm_position == "none" else "iden")
        )
        if mha_type == "relmha":
            self.mha = MultiHeadRelativeAttention(
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="mhsa",
            )
        else:
            self.mha = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="mhsa",
            )
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.post_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if norm_position == "post"
            else Identity(name="postiden" if norm_position == "none" else "iden")
        )
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual")

    def call(
        self,
        inputs,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        _inputs, relative_position_encoding, content_attention_bias, positional_attention_bias = inputs
        outputs = self.pre_norm(_inputs, training=training)
        outputs = self.mha(
            [outputs, outputs, outputs, relative_position_encoding, content_attention_bias, positional_attention_bias],
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
        )
        outputs = self.do(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        outputs = self.residual([_inputs, outputs], training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape, *_ = input_shape
        return tf.TensorShape(output_shape)


class ConvModule(Layer):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   conv1d(.)              # 2 * input_dim
      |    |
      |   glu(.)                  # input_dim
      |   depthwise_conv_1d(.)
      |   bnorm(.)
      |   swish(.)
      |    |
      |   conv1d(.)
      |   dropout(.)
      \   /
        +
        |
      output
    """

    def __init__(
        self,
        input_dim,
        kernel_size=32,
        dropout=0.0,
        padding="causal",
        scale_factor=2,
        residual_factor=1.0,
        norm_position="pre",
        use_group_conv=False,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conv_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        self.pre_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if norm_position == "pre"
            else Identity(name="preiden" if norm_position == "none" else "iden")
        )
        self.pw_conv_1 = Conv1D(
            filters=scale_factor * input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.glu = GLU(axis=-1, name="glu")
        if use_group_conv:
            self.dw_conv = Conv1D(
                filters=input_dim,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                groups=input_dim,
                name="dw_conv",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        else:
            self.dw_conv = DepthwiseConv1D(
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                name="dw_conv",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, synchronized=True
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name="swish")
        self.pw_conv_2 = Conv1D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(rate=dropout, name="dropout")
        self.post_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if norm_position == "post"
            else Identity(name="postiden" if norm_position == "none" else "iden")
        )
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual")

    def call(self, inputs, training=False):
        outputs = self.pre_norm(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs, training=training)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs, training=training)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        outputs = self.residual([inputs, outputs], training=training)
        return outputs


class ConformerBlock(Layer):
    r"""
    architecture::
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)
    """

    def __init__(
        self,
        input_dim,
        dropout=0.0,
        ffm_scale_factor=4,
        ffm_residual_factor=0.5,
        head_size=36,
        num_heads=4,
        mha_type="relmha",
        mhsam_residual_factor=1.0,
        kernel_size=32,
        padding="causal",
        convm_scale_factor=2,
        convm_residual_factor=1.0,
        convm_use_group_conv=False,
        module_norm_position="pre",
        block_norm_position="post",
        memory_length=None,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert block_norm_position in ("pre", "post", "none")
        self.pre_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if block_norm_position == "pre"
            else Identity(name="preiden" if block_norm_position == "none" else "iden")
        )
        self.ffm1 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            scale_factor=ffm_scale_factor,
            residual_factor=ffm_residual_factor,
            norm_position=module_norm_position,
            name="ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.mhsam = MHSAModule(
            dmodel=input_dim,
            head_size=head_size,
            num_heads=num_heads,
            residual_factor=mhsam_residual_factor,
            dropout=dropout,
            mha_type=mha_type,
            norm_position=module_norm_position,
            memory_length=memory_length,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="mhsa_module",
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            name="conv_module",
            padding=padding,
            scale_factor=convm_scale_factor,
            residual_factor=convm_residual_factor,
            norm_position=module_norm_position,
            use_group_conv=convm_use_group_conv,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            scale_factor=ffm_scale_factor,
            residual_factor=ffm_residual_factor,
            norm_position=module_norm_position,
            name="ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.post_norm = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
            if block_norm_position == "post"
            else Identity(name="postiden" if block_norm_position == "none" else "iden")
        )

    def call(
        self,
        inputs,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        inputs, relative_position_encoding, content_attention_bias, positional_attention_bias = inputs
        outputs = self.pre_norm(inputs, training=training)
        outputs = self.ffm1(outputs, training=training)
        outputs = self.mhsam(
            [outputs, relative_position_encoding, content_attention_bias, positional_attention_bias],
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
        )
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape, *_ = input_shape
        return tf.TensorShape(output_shape)


class ConformerEncoder(Layer):
    def __init__(
        self,
        subsampling,
        dmodel=144,
        num_blocks=16,
        mha_type="relmha",
        head_size=36,
        num_heads=4,
        kernel_size=32,
        padding="causal",
        interleave_relpe=True,
        use_attention_causal_mask=False,
        use_attention_auto_mask=True,
        ffm_scale_factor=4,
        ffm_residual_factor=0.5,
        mhsam_residual_factor=1.0,
        convm_scale_factor=2,
        convm_residual_factor=1.0,
        convm_use_group_conv=False,
        dropout=0.1,
        module_norm_position="pre",
        block_norm_position="post",
        memory_length=None,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert mha_type in ("relmha", "mha")
        self._dmodel = dmodel
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._num_blocks = num_blocks

        subsampling_name = subsampling.pop("type", None)
        if subsampling_name == "vgg":
            subsampling_class = VggSubsampling
        elif subsampling_name == "conv2d":
            subsampling_class = Conv2dSubsampling
        elif subsampling_name == "conv1d":
            subsampling_class = Conv1dSubsampling
        else:
            raise ValueError("subsampling must be either 'vgg', 'conv2d', 'conv1d'")

        self.conv_subsampling = subsampling_class(
            **subsampling,
            name="subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.time_reduction_factor = self.conv_subsampling.time_reduction_factor

        self.linear = tf.keras.layers.Dense(
            dmodel,
            name="linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")

        self._mha_type = mha_type
        self._num_heads = num_heads
        self._key_dim = head_size
        self._use_attention_causal_mask = use_attention_causal_mask
        self._use_attention_auto_mask = use_attention_auto_mask

        if self._mha_type == "relmha":
            self.relpe = RelativePositionalEncoding(interleave=interleave_relpe, memory_length=memory_length, name="relpe")
        else:
            self.relpe = PositionalEncoding(interleave=interleave_relpe, name="pe")

        self.conformer_blocks = [
            ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                ffm_scale_factor=ffm_scale_factor,
                ffm_residual_factor=ffm_residual_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                mhsam_residual_factor=mhsam_residual_factor,
                kernel_size=kernel_size,
                padding=padding,
                convm_scale_factor=convm_scale_factor,
                convm_residual_factor=convm_residual_factor,
                convm_use_group_conv=convm_use_group_conv,
                module_norm_position=module_norm_position,
                block_norm_position=block_norm_position,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            for i in range(self._num_blocks)
        ]

        if self._mha_type == "relmha":
            self.content_attention_bias = self.add_weight(
                name="content_attention_bias",
                shape=[self._num_heads, self._key_dim],
                trainable=True,
                initializer="zeros",
                regularizer=self._bias_regularizer,
            )
            self.positional_attention_bias = self.add_weight(
                name="positional_attention_bias",
                shape=[self._num_heads, self._key_dim],
                trainable=True,
                initializer="zeros",
                regularizer=self._bias_regularizer,
            )
        else:
            self.content_attention_bias, self.positional_attention_bias = None, None

    def get_states(self):
        return [block.mhsam.mha.get_states() for block in self.conformer_blocks]

    def reset_states(self, states=None):
        if states is None:
            states = [(None, None) for _ in range(self._num_blocks)]
        for i, memory_states in enumerate(states):
            self.conformer_blocks[i].mhsam.mha.reset_states(memory_states)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling([outputs, outputs_length], training=training)
        outputs = self.linear(outputs, training=training)
        outputs, relative_position_encoding = self.relpe(outputs, training=training)
        outputs = self.do(outputs, training=training)

        for cblock in self.conformer_blocks:
            outputs = cblock(
                [outputs, relative_position_encoding, self.content_attention_bias, self.positional_attention_bias],
                training=training,
                use_causal_mask=self._use_attention_causal_mask,
                use_auto_mask=self._use_attention_auto_mask,
            )

        return outputs, outputs_length

    def call_next(self, features, features_length, *args, **kwargs):
        """
        Recognize function for encoder network

        Parameters
        ----------
        features : tf.Tensor, shape [B, T, F, C]
        features_length : tf.Tensor, shape [B]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor], shape ([B, T, dmodel], [B], None)
            Outputs, outputs_length, new_states
        """
        with tf.name_scope(f"{self.name}_call_next"):
            outputs, outputs_length = self.call((features, features_length), training=False)
            return outputs, outputs_length, None

    def compute_output_shape(self, input_shape):
        outputs_shape, outputs_length_shape = self.conv_subsampling.compute_output_shape(input_shape)
        outputs_shape = list(outputs_shape)
        outputs_shape[-1] = self._dmodel
        return outputs_shape, outputs_length_shape
