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


from tensorflow_asr import keras
from tensorflow_asr.models.encoders.conformer import L2, ConformerEncoder
from tensorflow_asr.models.transducer.base_transducer import Transducer


@keras.utils.register_keras_serializable(package=__name__)
class Conformer(Transducer):
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
        prediction_label_encode_mode: str = "embedding",
        prediction_embed_dim: int = 512,
        prediction_num_rnns: int = 1,
        prediction_rnn_units: int = 320,
        prediction_rnn_type: str = "lstm",
        prediction_rnn_implementation: int = 2,
        prediction_rnn_unroll: bool = False,
        prediction_layer_norm: bool = True,
        prediction_projection_units: int = 0,
        prediction_trainable: bool = True,
        joint_dim: int = 1024,
        joint_activation: str = "tanh",
        prejoint_encoder_linear: bool = True,
        prejoint_prediction_linear: bool = True,
        postjoint_linear: bool = False,
        joint_mode: str = "add",
        joint_trainable: bool = True,
        kernel_regularizer=L2,
        bias_regularizer=None,
        activity_regularizer=None,
        recurrent_regularizer=None,
        name: str = "conformer",
        **kwargs,
    ):
        super().__init__(
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
            blank=blank,
            vocab_size=vocab_size,
            prediction_label_encoder_mode=prediction_label_encode_mode,
            prediction_embed_dim=prediction_embed_dim,
            prediction_num_rnns=prediction_num_rnns,
            prediction_rnn_units=prediction_rnn_units,
            prediction_rnn_type=prediction_rnn_type,
            prediction_rnn_implementation=prediction_rnn_implementation,
            prediction_rnn_unroll=prediction_rnn_unroll,
            prediction_layer_norm=prediction_layer_norm,
            prediction_projection_units=prediction_projection_units,
            prediction_trainable=prediction_trainable,
            joint_dim=joint_dim,
            joint_activation=joint_activation,
            prejoint_encoder_linear=prejoint_encoder_linear,
            prejoint_prediction_linear=prejoint_prediction_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            joint_trainable=joint_trainable,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            name=name,
            **kwargs,
        )
        self.dmodel = encoder_dmodel
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor

    def get_initial_encoder_states(self, batch_size=1):
        return self.encoder.get_initial_state(batch_size)
