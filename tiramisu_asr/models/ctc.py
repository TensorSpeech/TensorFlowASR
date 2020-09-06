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

import numpy as np
import tensorflow as tf

from ctc_decoders import ctc_greedy_decoder, ctc_beam_search_decoder

from ..featurizers.speech_featurizers import TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils.utils import shape_list


class CtcModel(tf.keras.Model):
    def __init__(self,
                 base_model: tf.keras.Model,
                 num_classes: int,
                 name="ctc_model",
                 **kwargs):
        super(CtcModel, self).__init__(name=name, **kwargs)
        self.base_model = base_model
        # Fully connected layer
        self.fc = tf.keras.layers.Dense(units=num_classes, activation="linear",
                                        use_bias=True, name=f"{name}_fc")

    def _build(self, input_shape):
        features = tf.keras.Input(input_shape, dtype=tf.float32)
        self(features, training=False)

    def summary(self, line_length=None, **kwargs):
        self.base_model.summary(line_length=line_length, **kwargs)
        super(CtcModel, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: TFSpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False, **kwargs):
        outputs = self.base_model(inputs, training=training)
        outputs = self.fc(outputs, training=training)
        return outputs

    def get_config(self):
        config = self.base_model.get_config()
        config.update(self.fc.get_config())
        return config

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self, features):
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob):
            return tf.numpy_function(self.perform_greedy,
                                     inp=[prob], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def perform_greedy(self, probs: np.ndarray):
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
        input_length = shape_list(features)[1]
        input_length = input_length // self.base_model.time_reduction_factor
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
    def recognize_beam(self, features, lm=False):
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob):
            return tf.numpy_function(self.perform_beam_search,
                                     inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def perform_beam_search(self,
                            probs: np.ndarray,
                            lm: bool = False):
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.vocab_array,
            beam_size=self.text_featurizer.decoder_config["beam_width"],
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        decoded = decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)

    def recognize_beam_tflite(self, signal):
        features = self.speech_featurizer.tf_extract(signal)
        features = tf.expand_dims(features, axis=0)
        input_length = shape_list(features)[1]
        input_length = input_length // self.base_model.time_reduction_factor
        input_length = tf.expand_dims(input_length, axis=0)
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)
        decoded = tf.keras.backend.ctc_decode(
            y_pred=probs, input_length=input_length, greedy=False,
            beam_width=self.text_featurizer.decoder_config["beam_width"]
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
