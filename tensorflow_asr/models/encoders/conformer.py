# pylint: disable=attribute-defined-outside-init
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
from keras.utils import tf_utils

from tensorflow_asr.models.activations.glu import GLU
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.layers.convolution import Conv1D, DepthwiseConv1D
from tensorflow_asr.models.layers.multihead_attention import MultiHeadAttention, MultiHeadRelativeAttention
from tensorflow_asr.models.layers.positional_encoding import PositionalEncoding, RelativePositionalEncoding
from tensorflow_asr.models.layers.residual import Residual
from tensorflow_asr.models.layers.subsampling import Conv1dSubsampling, Conv2dSubsampling, VggSubsampling
from tensorflow_asr.utils import data_util, shape_util

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
        self._norm_position = norm_position
        self.norm = (
            None
            if norm_position == "none"
            else tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
        )
        self.ffn1 = tf.keras.layers.Dense(
            scale_factor * input_dim,
            name="dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name="swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name="dropout_1")
        self.ffn2 = tf.keras.layers.Dense(input_dim, name="dense_2", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
        self.do2 = tf.keras.layers.Dropout(dropout, name="dropout_2")
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual")

    def call(self, inputs, training=False):
        outputs = self.norm(inputs, training=training) if self._norm_position == "pre" else inputs
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.norm(outputs, training=training) if self._norm_position == "post" else outputs
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
        self._norm_position = norm_position
        self.norm = (
            None
            if norm_position == "none"
            else tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
        )
        if mha_type == "relmha":
            self.mha = MultiHeadRelativeAttention(
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=tf.float32,  # for stable training
                name="mhsa",
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=tf.float32,  # for stable training
                name="mhsa",
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual")
        self.mha_type = mha_type

    def call(
        self,
        inputs,
        relative_position_encoding=None,
        content_attention_bias=None,
        positional_attention_bias=None,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        outputs = self.norm(inputs, training=training) if self._norm_position == "pre" else inputs
        mha_inputs = (
            dict(
                inputs=[outputs, outputs, outputs, relative_position_encoding],
                content_attention_bias=content_attention_bias,
                positional_attention_bias=positional_attention_bias,
            )
            if self.mha_type == "relmha"
            else dict(inputs=[outputs, outputs, outputs])
        )
        outputs = self.mha(
            **mha_inputs,
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
        )
        outputs = self.do(outputs, training=training)
        outputs = self.norm(outputs, training=training) if self._norm_position == "post" else outputs
        outputs = self.residual([inputs, outputs], training=training)
        return outputs


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
        depth_multiplier=1,
        padding="causal",
        scale_factor=2,
        residual_factor=1.0,
        dense_as_pointwise=False,
        depthwise_as_groupwise=False,
        norm_position="pre",
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conv_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        self._norm_position = norm_position
        self.norm = (
            None
            if norm_position == "none"
            else tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
        )
        if dense_as_pointwise:
            self.pw_conv_1 = tf.keras.layers.Dense(
                units=scale_factor * input_dim,
                name="pw_conv_1",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        else:
            self.pw_conv_1 = Conv1D(
                filters=scale_factor * input_dim,
                kernel_size=1,
                strides=1,
                padding=padding,
                name="pw_conv_1",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        self.glu = GLU(axis=-1, name="glu_activation")
        if depthwise_as_groupwise:
            self.dw_conv = Conv1D(
                filters=input_dim,
                kernel_size=kernel_size,
                strides=1,
                groups=input_dim,
                padding=padding,
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
                depth_multiplier=depth_multiplier,
                depthwise_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        self.bn = tf.keras.layers.BatchNormalization(name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name="swish_activation")
        if dense_as_pointwise:
            self.pw_conv_2 = tf.keras.layers.Dense(
                units=input_dim,
                name="pw_conv_2",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        else:
            self.pw_conv_2 = Conv1D(
                filters=input_dim,
                kernel_size=1,
                strides=1,
                padding=padding,
                name="pw_conv_2",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual")

    def call(self, inputs, training=False):
        outputs = self.norm(inputs, training=training) if self._norm_position == "pre" else inputs
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.norm(outputs, training=training) if self._norm_position == "post" else outputs
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
        depth_multiplier=1,
        padding="causal",
        dense_as_pointwise=False,
        depthwise_as_groupwise=False,
        convm_scale_factor=2,
        convm_residual_factor=1.0,
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
        self._norm_position = block_norm_position
        self.norm = (
            None
            if block_norm_position == "none"
            else tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer)
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
            depth_multiplier=depth_multiplier,
            padding=padding,
            dense_as_pointwise=dense_as_pointwise,
            depthwise_as_groupwise=depthwise_as_groupwise,
            scale_factor=convm_scale_factor,
            residual_factor=convm_residual_factor,
            norm_position=module_norm_position,
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

    def call(
        self,
        inputs,
        relative_position_encoding=None,
        content_attention_bias=None,
        positional_attention_bias=None,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        outputs = self.norm(inputs, training=training) if self._norm_position == "pre" else inputs
        outputs = self.ffm1(outputs, training=training)
        outputs = self.mhsam(
            outputs,
            relative_position_encoding=relative_position_encoding,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
        )
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.norm(outputs, training=training) if self._norm_position == "post" else outputs
        return outputs


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
        depth_multiplier=1,
        padding="causal",
        interleave_relpe=True,
        use_attention_causal_mask=False,
        use_attention_auto_mask=True,
        ffm_scale_factor=4,
        ffm_residual_factor=0.5,
        mhsam_residual_factor=1.0,
        convm_scale_factor=2,
        convm_residual_factor=1.0,
        dropout=0.1,
        dense_as_pointwise=False,
        depthwise_as_groupwise=False,
        module_norm_position="pre",
        block_norm_position="post",
        memory_length=None,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
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

        self.conformer_blocks = []
        for i in range(self._num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                ffm_scale_factor=ffm_scale_factor,
                ffm_residual_factor=ffm_residual_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                mhsam_residual_factor=mhsam_residual_factor,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                padding=padding,
                dense_as_pointwise=dense_as_pointwise,
                depthwise_as_groupwise=depthwise_as_groupwise,
                convm_scale_factor=convm_scale_factor,
                convm_residual_factor=convm_residual_factor,
                module_norm_position=module_norm_position,
                block_norm_position=block_norm_position,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            self.conformer_blocks.append(conformer_block)

        if self._mha_type == "relmha":
            self.content_attention_bias = self.add_weight(
                name="content_attention_bias",
                shape=[self._num_heads, self._key_dim],
                trainable=True,
                regularizer=self._bias_regularizer,
            )
            self.positional_attention_bias = self.add_weight(
                name="positional_attention_bias",
                shape=[self._num_heads, self._key_dim],
                trainable=True,
                regularizer=self._bias_regularizer,
            )
        else:
            self.content_attention_bias, self.positional_attention_bias = None, None

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling([outputs, outputs_length], training=training)
        outputs = self.linear(outputs, training=training)
        outputs, relative_position_encoding = self.relpe(outputs, training=training)
        outputs = self.do(outputs, training=training)

        for _, cblock in enumerate(self.conformer_blocks):
            outputs = cblock(
                outputs,
                relative_position_encoding=relative_position_encoding,
                content_attention_bias=self.content_attention_bias,
                positional_attention_bias=self.positional_attention_bias,
                training=training,
                use_causal_mask=self._use_attention_causal_mask,
                use_auto_mask=self._use_attention_auto_mask,
            )

        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        outputs_shape, outputs_length_shape = self.conv_subsampling.compute_output_shape(input_shape)
        outputs_shape = list(outputs_shape)
        outputs_shape[-1] = self._dmodel
        return outputs_shape, outputs_length_shape
