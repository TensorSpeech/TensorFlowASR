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

from typing import Dict, Union
import numpy as np
import tensorflow as tf

from ..base_model import BaseModel
from ...featurizers.speech_featurizers import TFSpeechFeaturizer
from ...featurizers.text_featurizers import TextFeaturizer
from ...utils import math_util, shape_util, data_util
from ...losses.ctc_loss import CtcLoss


class CtcModel(BaseModel):
    def __init__(self,
                 encoder: tf.keras.Model,
                 decoder: Union[tf.keras.Model, tf.keras.layers.Layer] = None,
                 vocabulary_size: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        if decoder is None:
            assert vocabulary_size is not None, "vocabulary_size must be set"
            self.decoder = tf.keras.layers.Dense(units=vocabulary_size, name=f"{self.name}_logits")
        else:
            self.decoder = decoder
        self.time_reduction_factor = 1

    def _build(self, input_shape, batch_size=None):
        inputs = tf.keras.Input(input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length
            ),
            training=False
        )

    def compile(self,
                optimizer,
                global_batch_size,
                blank=0,
                run_eagerly=None,
                **kwargs):
        loss = CtcLoss(blank=blank, global_batch_size=global_batch_size)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: TFSpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False, **kwargs):
        logits = self.encoder(inputs["inputs"], training=training, **kwargs)
        logits = self.decoder(logits, training=training, **kwargs)
        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        )

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self, inputs: Dict[str, tf.Tensor]):
        logits = self(inputs["inputs"], training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob): return tf.numpy_function(self._perform_greedy, inp=[prob], Tout=tf.string)

        return tf.map_fn(map_fn, probs, fn_output_signature=tf.TensorSpec([], dtype=tf.string))

    def _perform_greedy(self, probs: np.ndarray):
        from ctc_decoders import ctc_greedy_decoder
        decoded = ctc_greedy_decoder(probs, vocabulary=self.text_featurizer.vocab_array)
        return tf.convert_to_tensor(decoded, dtype=tf.string)

    def recognize_tflite(self, signal):
        """
        Function to convert to tflite using greedy decoding
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
        """
        features = self.speech_featurizer.tf_extract(signal)
        features = tf.expand_dims(features, axis=0)
        input_length = shape_util.shape_list(features)[1]
        input_length = math_util.get_reduced_length(input_length, self.time_reduction_factor)
        input_length = tf.expand_dims(input_length, axis=0)
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)
        decoded = tf.keras.backend.ctc_decode(
            y_pred=probs, input_length=input_length, greedy=True
        )
        decoded = tf.cast(decoded[0][0][0], dtype=tf.int32)
        transcript = self.text_featurizer.indices2upoints(decoded)
        return transcript

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self, inputs: Dict[str, tf.Tensor], lm: bool = False):
        logits = self(inputs["inputs"], training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob): return tf.numpy_function(self._perform_beam_search, inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def _perform_beam_search(self, probs: np.ndarray, lm: bool = False):
        from ctc_decoders import ctc_beam_search_decoder
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.vocab_array,
            beam_size=self.text_featurizer.decoder_config.beam_width,
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        decoded = decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)

    def recognize_beam_tflite(self, signal):
        """
        Function to convert to tflite using beam search decoding
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
        """
        features = self.speech_featurizer.tf_extract(signal)
        features = tf.expand_dims(features, axis=0)
        input_length = shape_util.shape_list(features)[1]
        input_length = math_util.get_reduced_length(input_length, self.time_reduction_factor)
        input_length = tf.expand_dims(input_length, axis=0)
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)
        decoded = tf.keras.backend.ctc_decode(
            y_pred=probs, input_length=input_length, greedy=False,
            beam_width=self.text_featurizer.decoder_config.beam_width
        )
        decoded = tf.cast(decoded[0][0][0], dtype=tf.int32)
        transcript = self.text_featurizer.indices2upoints(decoded)
        return transcript

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(self, greedy: bool = False):
        if greedy:
            return tf.function(
                self.recognize_tflite,
                input_signature=[
                    tf.TensorSpec([None], dtype=tf.float32)
                ]
            )
        return tf.function(
            self.recognize_beam_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32)
            ]
        )
