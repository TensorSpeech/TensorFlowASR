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

from tensorflow_asr import keras
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.ctc.base_ctc import CtcModel
from tensorflow_asr.models.encoders.conformer import L2, ConformerEncoder


@keras.utils.register_keras_serializable(package=__name__)
class ConformerDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        kernel_regularizer=L2,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self.vocab = keras.layers.Dense(
            units=vocab_size,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            name="logits",
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        logits, logits_length, *_ = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length, None

    def call_next(self, logits, logits_length, *args, **kwargs):
        return self((logits, logits_length), training=False)

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


@keras.utils.register_keras_serializable(package=__name__)
class Conformer(CtcModel):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        speech_config: dict,
        encoder_subsampling: dict,
        encoder_dmodel: int = 144,
        encoder_num_blocks: int = 16,
        encoder_head_size: int = 36,
        encoder_num_heads: int = 4,
        encoder_mha_type: str = "relmha",
        encoder_interleave_relpe: bool = True,
        encoder_use_attention_causal_mask: bool = False,
        encoder_use_attention_auto_mask: bool = True,
        encoder_kernel_size: int = 31,
        encoder_padding: str = "causal",
        encoder_ffm_scale_factor: int = 4,
        encoder_ffm_residual_factor: float = 0.5,
        encoder_mhsam_residual_factor: float = 1.0,
        encoder_mhsam_use_attention_bias: bool = False,
        encoder_mhsam_causal: bool = False,
        encoder_mhsam_flash_attention: bool = False,
        encoder_convm_scale_factor: int = 2,
        encoder_convm_residual_factor: float = 1.0,
        encoder_convm_use_group_conv: bool = False,
        encoder_convm_dw_norm_type: str = "batch",
        encoder_dropout: float = 0.1,
        encoder_module_norm_position: str = "pre",
        encoder_block_norm_position: str = "post",
        encoder_memory_length: int = None,
        encoder_history_size: int = None,
        encoder_chunk_size: int = None,
        encoder_trainable: bool = True,
        decoder_trainable: bool = True,
        kernel_regularizer=L2,
        bias_regularizer=None,
        activity_regularizer=None,
        name: str = "conformer",
        **kwargs,
    ):
        super().__init__(
            blank=blank,
            speech_config=speech_config,
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
                padding=encoder_padding,
                ffm_scale_factor=encoder_ffm_scale_factor,
                ffm_residual_factor=encoder_ffm_residual_factor,
                mhsam_residual_factor=encoder_mhsam_residual_factor,
                mhsam_use_attention_bias=encoder_mhsam_use_attention_bias,
                mhsam_causal=encoder_mhsam_causal,
                mhsam_flash_attention=encoder_mhsam_flash_attention,
                convm_scale_factor=encoder_convm_scale_factor,
                convm_residual_factor=encoder_convm_residual_factor,
                convm_use_group_conv=encoder_convm_use_group_conv,
                convm_dw_norm_type=encoder_convm_dw_norm_type,
                dropout=encoder_dropout,
                module_norm_position=encoder_module_norm_position,
                block_norm_position=encoder_block_norm_position,
                memory_length=encoder_memory_length,
                history_size=encoder_history_size,
                chunk_size=encoder_chunk_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
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

    def get_initial_encoder_states(self, batch_size=1):
        return self.encoder.get_initial_state(batch_size)
