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

from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.layers.multihead_attention import MultiHeadAttention, MultiHeadRelativeAttention
from tensorflow_asr.models.layers.positional_encoding import RelativeSinusoidalPositionalEncoding, SinusoidalPositionalEncoding
from tensorflow_asr.models.layers.residual import Residual
from tensorflow_asr.models.layers.subsampling import Conv1dSubsampling, Conv2dSubsampling, VggSubsampling


class Pointwiseffn(Layer):
    def __init__(
        self,
        dmodel,
        dff,
        activation="relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ffn1 = tf.keras.layers.Dense(
            units=dff,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ffn_1",
            dtype=self.dtype,
        )
        self.ffn2 = tf.keras.layers.Dense(
            units=dmodel,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ffn_2",
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        outputs = self.ffn1(inputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        return outputs


class TransformerBlock(Layer):
    def __init__(
        self,
        dmodel,
        dff,
        num_heads,
        head_size,
        mha_type="mha",
        relmha_causal=False,
        norm_position="post",
        residual_factor=1.0,
        pwffn_activation="relu",
        dropout=0.1,
        memory_length=None,
        use_attention_bias=False,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert norm_position in ("pre", "post", "none")
        assert mha_type in ("mha", "relmha")
        self._norm_position = norm_position
        self._mha_type = mha_type
        self.norm1 = (
            None
            if self._norm_position == "none"
            else tf.keras.layers.LayerNormalization(
                beta_regularizer=kernel_regularizer, gamma_regularizer=bias_regularizer, name="ln_1", dtype=self.dtype
            )
        )
        self.mha = (
            MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="mhsa",
                dtype=self.dtype,
            )
            if mha_type == "mha"
            else MultiHeadRelativeAttention(
                causal=relmha_causal,
                num_heads=num_heads,
                key_dim=head_size,
                output_shape=dmodel,
                memory_length=memory_length,
                use_attention_bias=use_attention_bias,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="mhsa",
                dtype=self.dtype,
            )
        )
        self.do1 = tf.keras.layers.Dropout(dropout, name="do_1", dtype=self.dtype)
        self.residual1 = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual_1", dtype=self.dtype)
        self.norm2 = (
            None
            if self._norm_position == "none"
            else tf.keras.layers.LayerNormalization(
                beta_regularizer=kernel_regularizer, gamma_regularizer=bias_regularizer, name="ln_2", dtype=self.dtype
            )
        )
        self.pwffn = Pointwiseffn(
            dmodel=dmodel,
            dff=dff,
            activation=pwffn_activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="pwffn",
            dtype=self.dtype,
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name="do_2", dtype=self.dtype)
        self.residual2 = Residual(factor=residual_factor, regularizer=bias_regularizer, name="residual_2", dtype=self.dtype)

    def call(
        self,
        inputs,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        original_outputs, caching, relative_position_encoding, content_attention_bias, positional_attention_bias = inputs
        outputs = self.norm1(original_outputs, training=training) if self._norm_position == "pre" else original_outputs
        outputs, caching = self.mha(
            [outputs, outputs, outputs, caching, relative_position_encoding, content_attention_bias, positional_attention_bias],
            training=training,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            use_auto_mask=use_auto_mask,
        )
        outputs = self.do1(outputs, training=training)
        outputs = self.norm1(outputs, training=training) if self._norm_position == "post" else outputs
        original_outputs = self.residual1([original_outputs, outputs], training=training)
        outputs = self.norm2(original_outputs, training=training) if self._norm_position == "pre" else original_outputs
        outputs = self.pwffn(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.norm2(outputs, training=training) if self._norm_position == "post" else outputs
        outputs = self.residual2([original_outputs, outputs], training=training)
        return outputs, caching

    def compute_output_shape(self, input_shape):
        output_shape, caching_shape, *_ = input_shape
        return output_shape, caching_shape


class TransformerEncoder(Layer):
    def __init__(
        self,
        subsampling,
        num_blocks=6,
        dmodel=512,
        dff=1024,
        num_heads=4,
        head_size=128,
        dropout=0.1,
        mha_type="mha",
        relmha_causal=False,
        norm_position="post",
        residual_factor=1.0,
        interleave_relpe=True,
        use_attention_causal_mask=False,
        use_attention_auto_mask=True,
        use_attention_bias=False,
        pwffn_activation="relu",
        memory_length=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="transformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._use_attention_causal_mask = use_attention_causal_mask
        self._use_attention_auto_mask = use_attention_auto_mask
        self._num_blocks = num_blocks
        self._dmodel = dmodel
        self._memory_length = memory_length

        subsampling_name = subsampling.pop("type", None)
        if subsampling_name == "vgg":
            subsampling_class = VggSubsampling
        elif subsampling_name == "conv2d":
            subsampling_class = Conv2dSubsampling
        elif subsampling_name == "conv1d":
            subsampling_class = Conv1dSubsampling
        else:
            raise ValueError("subsampling must be either 'vgg', 'conv2d', 'conv1d'")
        self.subsampling = subsampling_class(
            **subsampling,
            name="subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )
        self.time_reduction_factor = self.subsampling.time_reduction_factor
        self.linear = tf.keras.layers.Dense(
            units=dmodel,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="linear",
            dtype=self.dtype,
        )
        self.do = tf.keras.layers.Dropout(dropout, name="dropout", dtype=self.dtype)

        if mha_type == "relmha":
            self.relpe = RelativeSinusoidalPositionalEncoding(
                interleave=interleave_relpe,
                memory_length=memory_length,
                causal=relmha_causal,
                name="relpe",
                dtype=self.dtype,
            )
        else:
            self.relpe = SinusoidalPositionalEncoding(interleave=interleave_relpe, name="pe", dtype=self.dtype)

        self.blocks = [
            TransformerBlock(
                dmodel=dmodel,
                dff=dff,
                num_heads=num_heads,
                head_size=head_size,
                mha_type=mha_type,
                relmha_causal=relmha_causal,
                norm_position=norm_position,
                residual_factor=residual_factor,
                pwffn_activation=pwffn_activation,
                dropout=dropout,
                memory_length=memory_length,
                use_attention_bias=use_attention_bias,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
                dtype=self.dtype,
            )
            for i in range(self._num_blocks)
        ]

        if mha_type == "relmha" and not use_attention_bias:
            self.content_attention_bias = self.add_weight(
                name="content_attention_bias",
                shape=[num_heads, head_size],
                trainable=True,
                initializer="zeros",
                regularizer=bias_regularizer,
                dtype=self.variable_dtype,
            )
            self.positional_attention_bias = self.add_weight(
                name="positional_attention_bias",
                shape=[num_heads, head_size],
                trainable=True,
                initializer="zeros",
                regularizer=bias_regularizer,
                dtype=self.variable_dtype,
            )
        else:
            self.content_attention_bias, self.positional_attention_bias = None, None

    def reset_caching(self, batch_size):
        if self._memory_length is None:
            return None
        # fmt: off
        return [
            tf.zeros(shape=(batch_size, self._memory_length, self._dmodel), dtype=self.dtype)
            for _ in range(self._num_blocks)
        ]
        # fmt: on

    def call(self, inputs, training=False):
        outputs, outputs_length, caching = inputs
        outputs, outputs_length = self.subsampling([outputs, outputs_length], training=training)
        outputs = self.linear(outputs, training=training)
        outputs, relative_position_encoding = self.relpe([outputs, outputs_length], training=training)
        outputs = self.do(outputs, training=training)
        new_caching = None if self._memory_length is None else []
        for i, block in enumerate(self.blocks):
            outputs, new_cache = block(
                [
                    outputs,
                    None if caching is None else caching[i],
                    relative_position_encoding,
                    self.content_attention_bias,
                    self.positional_attention_bias,
                ],
                training=training,
                use_causal_mask=self._use_attention_causal_mask,
                use_auto_mask=self._use_attention_auto_mask,
            )
            if new_caching is not None:
                new_caching.append(new_cache)
        return outputs, outputs_length, new_caching

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
            outputs, outputs_length, _ = self.call((features, features_length, None), training=False)
            return outputs, outputs_length, None

    def compute_mask(self, inputs, mask=None):
        *outputs, caching = inputs
        return *self.subsampling.compute_mask(outputs, mask=mask), getattr(caching, "_keras_mask", None)

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape, caching_shape = input_shape
        output_shape, output_length_shape = self.subsampling.compute_output_shape((output_shape, output_length_shape))
        output_shape = self.linear.compute_output_shape(output_shape)
        output_shape, relative_position_encoding_shape = self.relpe.compute_output_shape((output_shape, output_length_shape))
        output_shape = self.do.compute_output_shape(output_shape)
        for block in self.blocks:
            output_shape, caching_shape = block.compute_output_shape((output_shape, caching_shape, relative_position_encoding_shape, None, None))
        return output_shape, output_length_shape, caching_shape
