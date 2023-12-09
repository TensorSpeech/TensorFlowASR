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
        self.vocab = tf.keras.layers.Dense(
            units=vocab_size,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="logits",
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def call_next(self, logits, logits_length, *args, **kwargs):
        outputs, outputs_length = self((logits, logits_length), training=False)
        return outputs, outputs_length, None

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.models.ctc")
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
        encoder_kernel_size: int = 32,
        encoder_padding: str = "causal",
        encoder_ffm_scale_factor: int = 4,
        encoder_ffm_residual_factor: float = 0.5,
        encoder_mhsam_residual_factor: float = 1.0,
        encoder_mhsam_use_attention_bias: bool = False,
        encoder_convm_scale_factor: int = 2,
        encoder_convm_residual_factor: float = 1.0,
        encoder_convm_use_group_conv: bool = False,
        encoder_dropout: float = 0.1,
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
                convm_scale_factor=encoder_convm_scale_factor,
                convm_residual_factor=encoder_convm_residual_factor,
                convm_use_group_conv=encoder_convm_use_group_conv,
                dropout=encoder_dropout,
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

    def reset_caching(self):
        return self.encoder.reset_caching(self._batch_size)

    def make(self, input_shape=[None], prediction_shape=[None], batch_size=None, **kwargs):
        self._batch_size = int(batch_size / self.distribute_strategy.num_replicas_in_sync)
        caching = (
            None
            if self.encoder._memory_length is None
            else [
                tf.keras.Input(shape=[self.encoder._memory_length, self.encoder._dmodel], batch_size=batch_size, dtype=tf.float32)
                for _ in range(self.encoder._num_blocks)
            ]
        )
        return super().make(input_shape, prediction_shape, batch_size, caching, **kwargs)
