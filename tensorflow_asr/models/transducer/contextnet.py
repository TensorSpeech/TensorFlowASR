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

from typing import Dict, List
import tensorflow as tf

from ..encoders.contextnet import ContextNetEncoder, L2
from .transducer import Transducer
from ...utils import math_util, data_util


class ContextNet(Transducer):
    def __init__(self,
                 vocabulary_size: int,
                 encoder_blocks: List[dict],
                 encoder_alpha: float = 0.5,
                 encoder_trainable: bool = True,
                 prediction_embed_dim: int = 512,
                 prediction_embed_dropout: int = 0,
                 prediction_num_rnns: int = 1,
                 prediction_rnn_units: int = 320,
                 prediction_rnn_type: str = "lstm",
                 prediction_rnn_implementation: int = 2,
                 prediction_layer_norm: bool = True,
                 prediction_projection_units: int = 0,
                 prediction_trainable: bool = True,
                 joint_dim: int = 1024,
                 joint_activation: str = "tanh",
                 prejoint_linear: bool = True,
                 postjoint_linear: bool = False,
                 joint_mode: str = "add",
                 joint_trainable: bool = True,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name: str = "contextnet",
                 **kwargs):
        super(ContextNet, self).__init__(
            encoder=ContextNetEncoder(
                blocks=encoder_blocks,
                alpha=encoder_alpha,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=encoder_trainable,
                name=f"{name}_encoder"
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=prediction_embed_dim,
            embed_dropout=prediction_embed_dropout,
            num_rnns=prediction_num_rnns,
            rnn_units=prediction_rnn_units,
            rnn_type=prediction_rnn_type,
            rnn_implementation=prediction_rnn_implementation,
            layer_norm=prediction_layer_norm,
            prediction_trainable=prediction_trainable,
            projection_units=prediction_projection_units,
            joint_dim=joint_dim,
            joint_activation=joint_activation,
            prejoint_linear=prejoint_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            joint_trainable=joint_trainable,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
            **kwargs
        )
        self.dmodel = self.encoder.blocks[-1].dmodel
        self.time_reduction_factor = 1
        for block in self.encoder.blocks: self.time_reduction_factor *= block.time_reduction_factor

    def call(self, inputs, training=False, **kwargs):
        enc = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=training, **kwargs)
        pred = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=training, **kwargs)
        logits = self.joint_net([enc, pred], training=training, **kwargs)
        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        )

    def encoder_inference(self, features: tf.Tensor, input_length: tf.Tensor):
        with tf.name_scope(f"{self.name}_encoder"):
            input_length = tf.expand_dims(tf.shape(features)[0], axis=0)
            outputs = tf.expand_dims(features, axis=0)
            outputs = self.encoder([outputs, input_length], training=False)
            return tf.squeeze(outputs, axis=0)

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self, inputs: Dict[str, tf.Tensor]):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of padded extracted features

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_greedy_batch(encoded=encoded, encoded_length=encoded_length)

    def recognize_tflite(self, signal, predicted, prediction_states):
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
            predicted: last predicted character with shape []
            prediction_states: lastest prediction states with shape [num_rnns, 1 or 2, 1, P]

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            predicted: last predicted character with shape []
            encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
            prediction_states: lastest prediction states with shape [num_rnns, 1 or 2, 1, P]
        """
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features, tf.shape(features)[0])
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, prediction_states)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, hypothesis.states

    def recognize_tflite_with_timestamp(self, signal, predicted, states):
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features, tf.shape(features)[0])
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states)
        indices = self.text_featurizer.normalize_indices(hypothesis.prediction)
        upoints = tf.gather_nd(self.text_featurizer.upoints, tf.expand_dims(indices, axis=-1))  # [None, max_subword_length]

        num_samples = tf.cast(tf.shape(signal)[0], dtype=tf.float32)
        total_time_reduction_factor = self.time_reduction_factor * self.speech_featurizer.frame_step

        stime = tf.range(0, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
        stime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

        etime = tf.range(total_time_reduction_factor, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
        etime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

        non_blank = tf.where(tf.not_equal(upoints, 0))
        non_blank_transcript = tf.gather_nd(upoints, non_blank)
        non_blank_stime = tf.gather_nd(tf.repeat(tf.expand_dims(stime, axis=-1), tf.shape(upoints)[-1], axis=-1), non_blank)
        non_blank_etime = tf.gather_nd(tf.repeat(tf.expand_dims(etime, axis=-1), tf.shape(upoints)[-1], axis=-1), non_blank)

        return non_blank_transcript, non_blank_stime, non_blank_etime, hypothesis.index, hypothesis.states

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self, inputs: Dict[str, tf.Tensor], lm: bool = False):
        """
        RNN Transducer Beam Search
        Args:
            features (tf.Tensor): a batch of padded extracted features
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_beam_search_batch(encoded=encoded, encoded_length=encoded_length, lm=lm)
