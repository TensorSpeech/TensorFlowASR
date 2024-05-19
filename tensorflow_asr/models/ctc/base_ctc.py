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

from tensorflow_asr import schemas
from tensorflow_asr.losses.ctc_loss import CtcLoss
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import layer_util


class CtcModel(BaseModel):
    def __init__(
        self,
        blank: int,
        speech_config: dict,
        encoder: tf.keras.layers.Layer,
        decoder: tf.keras.layers.Layer,
        **kwargs,
    ):
        super().__init__(speech_config=speech_config, **kwargs)
        self.blank = blank
        self.encoder = encoder
        self.decoder = decoder
        self.time_reduction_factor = 1

    def compile(self, optimizer, output_shapes=None, **kwargs):
        loss = CtcLoss(blank=self.blank, name="ctc_loss")
        return super().compile(loss, optimizer, **kwargs)

    def apply_gwn(self):
        if self.gwn_config:
            original_weights = {}
            if self.gwn_config.get("encoder_step") is not None and self.gwn_config.get("encoder_stddev") is not None:
                original_weights["encoder"] = tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["encoder_step"]),
                    lambda: layer_util.add_gwn(self.encoder.trainable_weights, stddev=self.gwn_config["encoder_stddev"]),
                    lambda: self.encoder.trainable_weights,
                )
            if self.gwn_config.get("decoder_step") is not None and self.gwn_config.get("decoder_stddev") is not None:
                original_weights["decoder"] = tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["decoder_step"]),
                    lambda: layer_util.add_gwn(self.decoder.trainable_weights, stddev=self.gwn_config["decoder_stddev"]),
                    lambda: self.decoder.trainable_weights,
                )
            return original_weights
        return {}

    def remove_gwn(self, original_weights):
        if self.gwn_config:
            if original_weights.get("encoder") is not None:
                tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["encoder_step"]),
                    lambda: layer_util.sub_gwn(original_weights["encoder"], self.encoder.trainable_weights),
                    lambda: None,
                )
            if original_weights.get("decoder") is not None:
                tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["decoder_step"]),
                    lambda: layer_util.sub_gwn(original_weights["decoder"], self.decoder.trainable_weights),
                    lambda: None,
                )

    def call(self, inputs: schemas.TrainInput, training=False):
        features, features_length = self.feature_extraction((inputs["inputs"], inputs["inputs_length"]), training=training)
        logits, logits_length, caching = self.encoder((features, features_length, inputs.get("caching")), training=training)
        logits, logits_length = self.decoder((logits, logits_length), training=training)
        return schemas.TrainOutput(
            logits=logits,
            logits_length=logits_length,
            caching=caching,
        )

    def call_next(
        self,
        features,
        features_length,
        previous_encoder_states=None,
        previous_decoder_states=None,
    ):
        outputs, outputs_length, next_encoder_states = self.encoder.call_next(features, features_length, previous_encoder_states)
        outputs, outputs_length, next_decoder_states = self.decoder.call_next(outputs, outputs_length, previous_decoder_states)
        return outputs, outputs_length, next_encoder_states, next_decoder_states

    def get_initial_tokens(self, batch_size=1):
        return super().get_initial_tokens(batch_size)

    def get_initial_encoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    def get_initial_decoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    # -------------------------------- GREEDY -------------------------------------

    def recognize(self, inputs: schemas.PredictInput, **kwargs):
        with tf.name_scope(f"{self.name}_recognize"):
            features, features_length = self.feature_extraction((inputs.inputs, inputs.inputs_length), training=False)
            (
                outputs,
                outputs_length,
                next_encoder_states,
                next_decoder_states,
            ) = self.call_next(features, features_length, inputs.previous_encoder_states, inputs.previous_decoder_states)
            tokens, _ = tf.nn.ctc_greedy_decoder(
                inputs=tf.transpose(outputs, perm=[1, 0, 2]),
                sequence_length=outputs_length,
                merge_repeated=True,
                blank_index=self.blank,
            )
            tokens = tf.sparse.to_dense(tokens[0])
            tokens = tf.cast(tokens, dtype=tf.int32)
            return schemas.PredictOutput(
                tokens=tokens,
                next_tokens=None,
                next_encoder_states=next_encoder_states,
                next_decoder_states=next_decoder_states,
            )

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(self, inputs: schemas.PredictInput, beam_width: int = 10, **kwargs):
        with tf.name_scope(f"{self.name}_recognize_beam"):
            features, features_length = self.feature_extraction((inputs.inputs, inputs.inputs_length), training=False)
            (
                outputs,
                outputs_length,
                next_encoder_states,
                next_decoder_states,
            ) = self.call_next(features, features_length, inputs.previous_encoder_states, inputs.previous_decoder_states)
            tokens, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(outputs, perm=[1, 0, 2]),
                sequence_length=outputs_length,
                beam_width=beam_width,
            )
            tokens = tf.sparse.to_dense(tokens[0])
            tokens = tf.cast(tokens, dtype=tf.int32)
            return schemas.PredictOutput(
                tokens=tokens,
                next_tokens=None,
                next_encoder_states=next_encoder_states,
                next_decoder_states=next_decoder_states,
            )
