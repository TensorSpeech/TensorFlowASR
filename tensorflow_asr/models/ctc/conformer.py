# Copyright 2022 Huy Le Nguyen (@nglehuy)
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
from tensorflow_asr.models.encoders.conformer import L2, ConformerEncoder


class ConformerDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self.vocab = tf.keras.layers.Conv1D(
            filters=vocab_size,
            kernel_size=1,
            strides=1,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="logits",
        )

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.models.ctc")
class Conformer(CtcModel):
    def __init__(
        self,
        vocab_size: int,
        encoder_subsampling: dict,
        encoder_dmodel: int = 144,
        encoder_num_blocks: int = 16,
        encoder_head_size: int = 36,
        encoder_num_heads: int = 4,
        encoder_mha_type: str = "relmha",
        encoder_interleave_relpe: bool = True,
        encoder_use_attention_causal_mask: bool = False,
        encoder_use_attention_auto_mask: bool = True,
        encoder_kernel_size: int = 32,
        encoder_depth_multiplier: int = 1,
        encoder_padding: str = "causal",
        encoder_ffm_scale_factor: int = 4,
        encoder_ffm_residual_factor: float = 0.5,
        encoder_mhsam_residual_factor: float = 1.0,
        encoder_convm_scale_factor: int = 2,
        encoder_convm_residual_factor: float = 1.0,
        encoder_dropout: float = 0.1,
        encoder_depthwise_as_groupwise: bool = False,
        encoder_module_norm_position: str = "pre",
        encoder_block_norm_position: str = "post",
        encoder_memory_length: int = None,
        encoder_trainable: bool = True,
        decoder_trainable: bool = True,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name: str = "conformer",
        **kwargs,
    ):
        super().__init__(
            encoder=ConformerEncoder(
                subsampling=encoder_subsampling,
                dmodel=encoder_dmodel,
                num_blocks=encoder_num_blocks,
                head_size=encoder_head_size,
                num_heads=encoder_num_heads,
                mha_type=encoder_mha_type,
                interleave_relpe=encoder_interleave_relpe,
                use_attention_causal_mask=encoder_use_attention_causal_mask,
                use_attention_auto_mask=encoder_use_attention_auto_mask,
                kernel_size=encoder_kernel_size,
                depth_multiplier=encoder_depth_multiplier,
                padding=encoder_padding,
                ffm_scale_factor=encoder_ffm_scale_factor,
                ffm_residual_factor=encoder_ffm_residual_factor,
                mhsam_residual_factor=encoder_mhsam_residual_factor,
                convm_scale_factor=encoder_convm_scale_factor,
                convm_residual_factor=encoder_convm_residual_factor,
                dropout=encoder_dropout,
                depthwise_as_groupwise=encoder_depthwise_as_groupwise,
                module_norm_position=encoder_module_norm_position,
                block_norm_position=encoder_block_norm_position,
                memory_length=encoder_memory_length,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=encoder_trainable,
                name="encoder",
            ),
            decoder=ConformerDecoder(
                vocab_size=vocab_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=decoder_trainable,
                name="decoder",
            ),
            name=name,
            **kwargs,
        )
        self.dmodel = encoder_dmodel
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor
