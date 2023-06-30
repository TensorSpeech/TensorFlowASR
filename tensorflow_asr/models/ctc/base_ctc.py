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

    def compile(self, optimizer, **kwargs):
        loss = CtcLoss(blank=self.blank, name="ctc_loss")
        return super().compile(loss, optimizer, **kwargs)

    def apply_gwn(self):
        if self.gwn_config:
            original_weights = {}
            if self.gwn_config.get("encoder_step") is not None and self.gwn_config.get("encoder_stddev") is not None:
                original_weights["encoder"] = tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.gwn_config["encoder_step"]),
                    lambda: layer_util.add_gwn(self.encoder.trainable_weights, stddev=self.gwn_config["encoder_stddev"]),
                    lambda: self.encoder.trainable_weights,
                )
            if self.gwn_config.get("decoder_step") is not None and self.gwn_config.get("decoder_stddev") is not None:
                original_weights["decoder"] = tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.gwn_config["decoder_step"]),
                    lambda: layer_util.add_gwn(self.decoder.trainable_weights, stddev=self.gwn_config["decoder_stddev"]),
                    lambda: self.decoder.trainable_weights,
                )
            return original_weights
        return {}

    def remove_gwn(self, original_weights):
        if self.gwn_config:
            if original_weights.get("encoder") is not None:
                tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.gwn_config["encoder_step"]),
                    lambda: layer_util.sub_gwn(original_weights["encoder"], self.encoder.trainable_weights),
                    lambda: None,
                )
            if original_weights.get("decoder") is not None:
                tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.gwn_config["decoder_step"]),
                    lambda: layer_util.sub_gwn(original_weights["decoder"], self.decoder.trainable_weights),
                    lambda: None,
                )

    def call_logits(self, features, features_length, *args, training=False):
        logits, logits_length = self.encoder((features, features_length), training=training)
        logits, logits_length = self.decoder((logits, logits_length), training=training)
        return logits, logits_length

    # -------------------------------- GREEDY -------------------------------------

    def recognize(self, inputs: tf.Tensor, inputs_length: tf.Tensor, **kwargs):
        with tf.name_scope(f"{self.name}_recognize"):
            features, features_length = self.feature_extraction((inputs, inputs_length), training=False)
            logits, logits_length = self.call_logits(features, features_length, training=False)
            tokens, _ = tf.nn.ctc_greedy_decoder(
                inputs=tf.transpose(logits, perm=[1, 0, 2]),
                sequence_length=logits_length,
                merge_repeated=True,
                blank_index=self.blank,
            )
            tokens = tf.reshape(tokens[0].values, tokens[0].dense_shape)
            tokens = tf.cast(tokens, dtype=tf.int32)
            return tokens

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(self, inputs: tf.Tensor, inputs_length: tf.Tensor, beam_width: int = 10, **kwargs):
        with tf.name_scope(f"{self.name}_recognize_beam"):
            features, features_length = self.feature_extraction((inputs, inputs_length), training=False)
            logits, logits_length = self.call_logits(features, features_length, training=False)
            tokens, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(logits, perm=[1, 0, 2]),
                sequence_length=logits_length,
                beam_width=beam_width,
            )
            tokens = tf.reshape(tokens[0].values, tokens[0].dense_shape)
            tokens = tf.cast(tokens, dtype=tf.int32)
            return tokens

    # -------------------------------- TFLITE -------------------------------------

    # def make_tflite_function(self, greedy: bool = False):
    #     if greedy:
    #         return tf.function(
    #             self.recognize_tflite,
    #             input_signature=[tf.TensorSpec([None], dtype=tf.float32)],
    #         )
    #     return tf.function(
    #         self.recognize_beam_tflite,
    #         input_signature=[tf.TensorSpec([None], dtype=tf.float32)],
    #     )
