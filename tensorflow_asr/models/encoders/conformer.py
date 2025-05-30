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
"""http://arxiv.org/abs/2005.08100"""

from tensorflow_asr import keras, tf
from tensorflow_asr.models.activations.glu import GLU
from tensorflow_asr.models.layers.convolution import Conv1D, DepthwiseConv1D
from tensorflow_asr.models.layers.general import Activation, Dropout, Identity
from tensorflow_asr.models.layers.multihead_attention import MultiHeadAttention, MultiHeadRelativeAttention
from tensorflow_asr.models.layers.positional_encoding import RelativeSinusoidalPositionalEncoding, SinusoidalPositionalEncoding
from tensorflow_asr.models.layers.residual import Residual
from tensorflow_asr.utils import data_util

L2 = keras.regularizers.l2(1e-6)


@keras.utils.register_keras_serializable(package=__name__)
class FFModule(keras.Model):
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
        bias_regularizer=None,
        name="ff_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        self.pre_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if norm_position == "pre"
            else Identity(name="preiden" if norm_position == "none" else "iden", dtype=self.dtype)
        )
        self.ffn1 = keras.layers.Dense(
            units=scale_factor * input_dim,
            name="dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation="swish",
            dtype=self.dtype,
        )
        self.do1 = Dropout(rate=dropout, name="dropout_1", dtype=self.dtype)
        self.ffn2 = keras.layers.Dense(
            units=input_dim,
            name="dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )
        self.do2 = Dropout(rate=dropout, name="dropout_2", dtype=self.dtype)
        self.post_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if norm_position == "post"
            else Identity(name="postiden" if norm_position == "none" else "iden", dtype=self.dtype)
        )
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual", dtype=self.dtype)

    def call(self, inputs, training=False):
        outputs = self.pre_norm(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        outputs = self.residual((inputs, outputs), training=training)
        return outputs


@keras.utils.register_keras_serializable(package=__name__)
class MHSAModule(keras.Model):
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
        relmha_causal=False,
        flash_attention=None,
        norm_position="pre",
        memory_length=None,
        history_size=None,
        chunk_size=None,
        use_attention_bias=False,
        kernel_regularizer=L2,
        bias_regularizer=None,
        activity_regularizer=None,
        name="mhsa_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        assert mha_type in ("relmha", "mha")
        self.pre_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if norm_position == "pre"
            else Identity(name="preiden" if norm_position == "none" else "iden", dtype=self.dtype)
        )
        if mha_type == "relmha":
            self.mha = MultiHeadRelativeAttention(
                causal=relmha_causal,
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                history_size=history_size,
                chunk_size=chunk_size,
                flash_attention=flash_attention,
                use_attention_bias=use_attention_bias,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                name="mhsa",
                dtype=self.dtype,
            )
        else:
            self.mha = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                history_size=history_size,
                chunk_size=chunk_size,
                flash_attention=flash_attention,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                name="mhsa",
                dtype=self.dtype,
            )
        self.do = Dropout(dropout, name="dropout", dtype=self.dtype)
        self.post_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if norm_position == "post"
            else Identity(name="postiden" if norm_position == "none" else "iden", dtype=self.dtype)
        )
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual", dtype=self.dtype)

    def get_initial_state(self, batch_size: int):
        return self.mha.get_initial_state(batch_size)

    def call(
        self,
        inputs,
        content_attention_bias=None,
        positional_attention_bias=None,
        initial_state=None,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
        return_states=False,
    ):
        _inputs, relative_position_encoding = inputs
        outputs = self.pre_norm(_inputs, training=training)
        outputs, *states = self.mha(
            [outputs, outputs, outputs, relative_position_encoding],
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            initial_state=initial_state,
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
            return_states=return_states,
        )
        outputs = self.do(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        outputs = self.residual((_inputs, outputs), training=training)
        if return_states:
            return [outputs] + states
        return [outputs]


@keras.utils.register_keras_serializable(package=__name__)
class ConvModule(keras.Model):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   conv1d(.)               # 2 * input_dim
      |    |
      |   glu(.)                  # input_dim
      |   depthwise_conv_1d(.)
      |   norm(.)                 # batch or layer
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
        dw_norm_type="batch",
        use_group_conv=False,
        kernel_regularizer=L2,
        bias_regularizer=None,
        name="conv_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert norm_position in ("pre", "post", "none")
        assert dw_norm_type in ("batch", "layer")
        self.pre_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if norm_position == "pre"
            else Identity(name="preiden" if norm_position == "none" else "iden", dtype=self.dtype)
        )
        self.pw_conv_1 = Conv1D(
            filters=scale_factor * input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )
        self.glu = GLU(axis=-1, name="glu", dtype=self.dtype)
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
                dtype=self.dtype,
            )
        else:
            self.dw_conv = DepthwiseConv1D(
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                name="dw_conv",
                depthwise_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=self.dtype,
            )
        self.dw_norm = (
            keras.layers.BatchNormalization(
                name="dw_bn",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                synchronized=True,
                dtype=self.dtype,
            )
            if dw_norm_type == "batch"
            else keras.layers.LayerNormalization(
                name="dw_ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
        )
        self.swish = Activation(tf.nn.swish, name="swish", dtype=self.dtype)
        self.pw_conv_2 = Conv1D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )
        self.do = Dropout(rate=dropout, name="dropout", dtype=self.dtype)
        self.post_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if norm_position == "post"
            else Identity(name="postiden" if norm_position == "none" else "iden", dtype=self.dtype)
        )
        self.residual = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual", dtype=self.dtype)

    def call(self, inputs, training=False):
        outputs = self.pre_norm(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs, training=training)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.dw_norm(outputs, training=training)
        outputs = self.swish(outputs, training=training)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        outputs = self.residual((inputs, outputs), training=training)
        return outputs


@keras.utils.register_keras_serializable(package=__name__)
class ConformerBlock(keras.Model):
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
        mhsam_use_attention_bias=False,
        mhsam_causal=False,
        mhsam_flash_attention=None,
        kernel_size=32,
        padding="causal",
        convm_scale_factor=2,
        convm_residual_factor=1.0,
        convm_use_group_conv=False,
        convm_dw_norm_type="batch",
        module_norm_position="pre",
        block_norm_position="post",
        memory_length=None,
        history_size=None,
        chunk_size=None,
        kernel_regularizer=L2,
        bias_regularizer=None,
        activity_regularizer=None,
        name="conformer_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert block_norm_position in ("pre", "post", "none")
        self.pre_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if block_norm_position == "pre"
            else Identity(name="preiden" if block_norm_position == "none" else "iden", dtype=self.dtype)
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
            dtype=self.dtype,
        )
        self.mhsam = MHSAModule(
            dmodel=input_dim,
            head_size=head_size,
            num_heads=num_heads,
            residual_factor=mhsam_residual_factor,
            use_attention_bias=mhsam_use_attention_bias,
            dropout=dropout,
            mha_type=mha_type,
            relmha_causal=mhsam_causal,
            norm_position=module_norm_position,
            memory_length=memory_length,
            history_size=history_size,
            chunk_size=chunk_size,
            flash_attention=mhsam_flash_attention,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            name="mhsa_module",
            dtype=self.dtype,
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
            dw_norm_type=convm_dw_norm_type,
            use_group_conv=convm_use_group_conv,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
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
            dtype=self.dtype,
        )
        self.post_norm = (
            keras.layers.LayerNormalization(
                name="ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )
            if block_norm_position == "post"
            else Identity(name="postiden" if block_norm_position == "none" else "iden", dtype=self.dtype)
        )

    def get_initial_state(self, batch_size: int):
        return self.mhsam.get_initial_state(batch_size)

    def call(
        self,
        inputs,
        content_attention_bias=None,
        positional_attention_bias=None,
        initial_state=None,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
        return_states=False,
    ):
        _inputs, relative_position_encoding = inputs
        outputs = self.pre_norm(_inputs, training=training)
        outputs = self.ffm1(outputs, training=training)
        outputs, *states = self.mhsam(
            [outputs, relative_position_encoding],
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            initial_state=initial_state,
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
            return_states=return_states,
        )
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.post_norm(outputs, training=training)
        if return_states:
            return [outputs] + states
        return [outputs]


@keras.utils.register_keras_serializable(package=__name__)
class ConformerEncoder(keras.Model):
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
        mhsam_use_attention_bias=False,
        mhsam_causal=False,
        mhsam_flash_attention=None,
        convm_scale_factor=2,
        convm_residual_factor=1.0,
        convm_use_group_conv=False,
        convm_dw_norm_type="batch",
        dropout=0.1,
        module_norm_position="pre",
        block_norm_position="post",
        memory_length=None,
        history_size=None,
        chunk_size=None,
        kernel_regularizer=L2,
        bias_regularizer=None,
        activity_regularizer=None,
        name="conformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert mha_type in ("relmha", "mha")
        self._dmodel = dmodel
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._num_blocks = num_blocks

        self.conv_subsampling = keras.utils.get_registered_object(name=subsampling["class_name"])(
            **subsampling["config"],
            name="subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )
        self.time_reduction_factor = self.conv_subsampling.time_reduction_factor

        self.linear = keras.layers.Dense(
            dmodel, name="linear", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.do = Dropout(dropout, name="dropout", dtype=self.dtype)

        self._mha_type = mha_type
        self._num_heads = num_heads
        self._key_dim = head_size
        self._memory_length = memory_length
        self._use_attention_causal_mask = use_attention_causal_mask
        self._use_attention_auto_mask = use_attention_auto_mask

        if self._mha_type == "relmha":
            self.relpe = RelativeSinusoidalPositionalEncoding(
                interleave=interleave_relpe,
                memory_length=memory_length,
                causal=mhsam_causal,
                name="relpe",
                dtype=self.dtype,
            )
        else:
            self.relpe = SinusoidalPositionalEncoding(interleave=interleave_relpe, name="pe", dtype=self.dtype)

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
                mhsam_use_attention_bias=mhsam_use_attention_bias,
                mhsam_causal=mhsam_causal,
                mhsam_flash_attention=mhsam_flash_attention,
                kernel_size=kernel_size,
                padding=padding,
                convm_scale_factor=convm_scale_factor,
                convm_residual_factor=convm_residual_factor,
                convm_use_group_conv=convm_use_group_conv,
                convm_dw_norm_type=convm_dw_norm_type,
                module_norm_position=module_norm_position,
                block_norm_position=block_norm_position,
                memory_length=memory_length,
                history_size=history_size,
                chunk_size=chunk_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                name=f"block_{i}",
                dtype=self.dtype,
            )
            for i in range(self._num_blocks)
        ]

        if self._mha_type == "relmha" and not mhsam_use_attention_bias:
            self.content_attention_bias = self.add_weight(
                name="content_attention_bias",
                shape=[self._num_heads, self._key_dim],
                trainable=True,
                initializer="zeros",
                regularizer=self._bias_regularizer,
                dtype=self.variable_dtype,
            )
            self.positional_attention_bias = self.add_weight(
                name="positional_attention_bias",
                shape=[self._num_heads, self._key_dim],
                trainable=True,
                initializer="zeros",
                regularizer=self._bias_regularizer,
                dtype=self.variable_dtype,
            )
        else:
            self.content_attention_bias, self.positional_attention_bias = None, None

    def get_initial_state(self, batch_size: int):
        states = [block.get_initial_state(batch_size) for block in self.conformer_blocks]
        states = [s for s in states if s is not None]
        return states

    def call(
        self,
        inputs,
        initial_state=None,
        training=False,
        return_states=False,
    ):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling((outputs, outputs_length), training=training)
        outputs = self.linear(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs, relative_position_encoding = self.relpe((outputs, outputs_length), training=training)
        states = None if self._memory_length is None else []
        for i, cblock in enumerate(self.conformer_blocks):
            outputs, *_states = cblock(
                (outputs, relative_position_encoding),
                content_attention_bias=self.content_attention_bias,
                positional_attention_bias=self.positional_attention_bias,
                initial_state=data_util.get(initial_state, i, None),
                training=training,
                use_causal_mask=self._use_attention_causal_mask,
                use_auto_mask=self._use_attention_auto_mask,
                return_states=return_states,
            )
            if not states:
                continue
            states.extend(_states)
        if return_states:
            return outputs, outputs_length, states
        return outputs, outputs_length

    def call_next(self, features, features_length, previous_encoder_states, *args, **kwargs):
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
            return self((features, features_length), initial_state=previous_encoder_states, training=False, return_states=True)

    def compute_mask(self, inputs, mask=None):
        return self.conv_subsampling.compute_mask(inputs, mask=mask)
