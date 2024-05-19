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
""" http://arxiv.org/abs/1811.06621 """

import tensorflow as tf

from tensorflow_asr.models.encoders.rnnt import RnnTransducerEncoder
from tensorflow_asr.models.transducer.base_transducer import Transducer


@tf.keras.utils.register_keras_serializable("tensorflow_asr.models.transducer")
class RnnTransducer(Transducer):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        speech_config: dict,
        encoder_reduction_factors: list = [6, 0, 0, 0, 0, 0, 0, 0],
        encoder_dmodel: int = 640,
        encoder_nlayers: int = 8,
        encoder_rnn_type: str = "lstm",
        encoder_rnn_units: int = 2048,
        encoder_rnn_unroll: bool = False,
        encoder_layer_norm: bool = False,
        encoder_trainable: bool = True,
        prediction_label_encode_mode: str = "embedding",
        prediction_embed_dim: int = 320,
        prediction_num_rnns: int = 2,
        prediction_rnn_units: int = 2048,
        prediction_rnn_type: str = "lstm",
        prediction_rnn_implementation: int = 2,
        prediction_rnn_unroll: bool = False,
        prediction_layer_norm: bool = False,
        prediction_projection_units: int = 640,
        prediction_trainable: bool = True,
        joint_dim: int = 640,
        joint_activation: str = "tanh",
        prejoint_encoder_linear: bool = True,
        prejoint_prediction_linear: bool = True,
        postjoint_linear: bool = False,
        joint_mode: str = "add",
        joint_trainable: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="rnn_transducer",
        **kwargs,
    ):
        super().__init__(
            speech_config=speech_config,
            encoder=RnnTransducerEncoder(
                reduction_factors=encoder_reduction_factors,
                dmodel=encoder_dmodel,
                nlayers=encoder_nlayers,
                rnn_type=encoder_rnn_type,
                rnn_units=encoder_rnn_units,
                rnn_unroll=encoder_rnn_unroll,
                layer_norm=encoder_layer_norm,
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
            prediction_layer_norm=prediction_layer_norm,
            prediction_rnn_implementation=prediction_rnn_implementation,
            prediction_rnn_unroll=prediction_rnn_unroll,
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
            name=name,
            **kwargs,
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor
        self.dmodel = encoder_dmodel

    def get_initial_encoder_states(self, batch_size=1):
        return self.encoder.get_initial_state(batch_size)
