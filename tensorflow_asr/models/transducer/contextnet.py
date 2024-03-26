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

from typing import List

import tensorflow as tf

from tensorflow_asr.models.encoders.contextnet import L2, ContextNetEncoder
from tensorflow_asr.models.transducer.base_transducer import Transducer


@tf.keras.utils.register_keras_serializable("tensorflow_asr.models.transducer")
class ContextNet(Transducer):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        speech_config: dict,
        encoder_blocks: List[dict],
        encoder_alpha: float = 0.5,
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
        bias_regularizer=L2,
        name: str = "contextnet",
        **kwargs,
    ):
        super().__init__(
            speech_config=speech_config,
            encoder=ContextNetEncoder(
                blocks=encoder_blocks,
                alpha=encoder_alpha,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
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
            prediction_trainable=prediction_trainable,
            prediction_projection_units=prediction_projection_units,
            joint_dim=joint_dim,
            joint_activation=joint_activation,
            prejoint_encoder_linear=prejoint_encoder_linear,
            prejoint_prediction_linear=prejoint_prediction_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            joint_trainable=joint_trainable,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
            **kwargs,
        )
        self.dmodel = self.encoder.blocks[-1].dmodel
        self.time_reduction_factor = 1
        for block in self.encoder.blocks:
            self.time_reduction_factor *= block.time_reduction_factor
