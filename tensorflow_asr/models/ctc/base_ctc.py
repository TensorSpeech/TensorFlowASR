# Copyright 2020 Huy Le Nguyen (@usimarit)
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

from typing import Dict

import numpy as np
import tensorflow as tf

from tensorflow_asr.losses.ctc_loss import CtcLoss
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import data_util, math_util, shape_util


class CtcModel(BaseModel):
    def __init__(
        self,
        encoder: tf.keras.layers.Layer,
        decoder: tf.keras.layers.Layer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.time_reduction_factor = 1

    def make(self, input_shape, batch_size=None):
        inputs = tf.keras.Input(input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length,
            ),
            training=False,
        )

    def compile(
        self,
        optimizer,
        blank=0,
        run_eagerly=None,
        mxp=True,
        ga_steps=None,
        **kwargs,
    ):
        loss = CtcLoss(blank=blank)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, mxp=mxp, ga_steps=ga_steps, **kwargs)

    def call(self, inputs, training=False):
        logits, logits_length = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=training)
        logits, logits_length = self.decoder([logits, logits_length], training=training)
        logits = tf.cast(logits, tf.float32)  # always cast the output as float32 for stable mxp training
        return data_util.create_logits(logits=logits, logits_length=logits_length)

    # -------------------------------- GREEDY -------------------------------------

    def recognize(self, inputs: Dict[str, tf.Tensor]):
        outputs = self(inputs, training=False)
        decoded = self._perform_greedy(encoded=outputs["logits"], encoded_length=outputs["logits_length"])
        return self.text_featurizer.iextract(decoded)

    def _perform_greedy(self, encoded, encoded_length):
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(encoded, perm=[1, 0, 2]),
            sequence_length=encoded_length,
            merge_repeated=True,
            blank_index=self.text_featurizer.blank,
        )
        decoded = tf.reshape(decoded[0].values, decoded[0].dense_shape)
        return tf.cast(decoded, dtype=tf.int32)

    def recognize_tflite(self, signal):
        """
        Function to convert to tflite using greedy decoding
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
        """
        inputs = self.speech_featurizer.tf_extract(signal)
        inputs = tf.expand_dims(inputs, axis=0)
        inputs_length = shape_util.shape_list(inputs)[1]
        inputs_length = math_util.get_reduced_length(inputs_length, self.time_reduction_factor)
        inputs_length = tf.expand_dims(inputs_length, axis=0)
        outputs = self(data_util.create_inputs(inputs=inputs, inputs_length=inputs_length))
        decoded = self._perform_greedy(encoded=outputs["logits"], encoded_length=outputs["logits_length"])
        transcript = self.text_featurizer.indices2upoints(decoded)
        return transcript

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(self, inputs: Dict[str, tf.Tensor], lm: bool = False):
        logits = self(inputs, training=False)
        decoded = self._perform_beam_search(encoded=logits["logits"], encoded_length=logits["logits_length"])
        return self.text_featurizer.iextract(decoded)

    def _perform_beam_search(self, encoded: np.ndarray, encoded_length):
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf.transpose(encoded, perm=[1, 0, 2]),
            sequence_length=encoded_length,
            beam_width=self.text_featurizer.decoder_config.beam_width,
        )
        decoded = tf.reshape(decoded[0].values, decoded[0].dense_shape)
        return tf.cast(decoded, dtype=tf.int32)

    def recognize_beam_tflite(self, signal):
        """
        Function to convert to tflite using beam search decoding
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
        """
        inputs = self.speech_featurizer.tf_extract(signal)
        inputs = tf.expand_dims(inputs, axis=0)
        inputs_length = shape_util.shape_list(inputs)[1]
        inputs_length = math_util.get_reduced_length(inputs_length, self.time_reduction_factor)
        inputs_length = tf.expand_dims(inputs_length, axis=0)
        outputs = self(data_util.create_inputs(inputs=inputs, inputs_length=inputs_length))
        decoded = self._perform_beam_search(encoded=outputs["logits"], encoded_length=outputs["logits_length"])
        transcript = self.text_featurizer.indices2upoints(decoded)
        return transcript

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(self, greedy: bool = False):
        if greedy:
            return tf.function(
                self.recognize_tflite,
                input_signature=[tf.TensorSpec([None], dtype=tf.float32)],
            )
        return tf.function(
            self.recognize_beam_tflite,
            input_signature=[tf.TensorSpec([None], dtype=tf.float32)],
        )
