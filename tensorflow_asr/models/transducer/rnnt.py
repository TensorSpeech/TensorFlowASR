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

    # def encoder_inference(self, features: tf.Tensor, states: tf.Tensor):
    #     """Infer function for encoder (or encoders)

    #     Args:
    #         features (tf.Tensor): features with shape [T, F, C]
    #         states (tf.Tensor): previous states of encoders with shape [num_rnns, 1 or 2, 1, P]

    #     Returns:
    #         tf.Tensor: output of encoders with shape [T, E]
    #         tf.Tensor: states of encoders with shape [num_rnns, 1 or 2, 1, P]
    #     """
    #     with tf.name_scope("encoder"):
    #         outputs = tf.expand_dims(features, axis=0)
    #         outputs, new_states = self.encoder.recognize(outputs, states)
    #         return tf.squeeze(outputs, axis=0), new_states

    # # -------------------------------- GREEDY -------------------------------------

    # def recognize_tflite(self, signal, predicted, encoder_states, prediction_states):
    #     """
    #     Function to convert to tflite using greedy decoding (default streaming mode)
    #     Args:
    #         signal: tf.Tensor with shape [None] indicating a single audio signal
    #         predicted: last predicted character with shape []
    #         encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
    #         prediction_states: lastest prediction states with shape [num_rnns, 1 or 2, 1, P]

    #     Return:
    #         transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
    #         predicted: last predicted character with shape []
    #         encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
    #         prediction_states: lastest prediction states with shape [num_rnns, 1 or 2, 1, P]
    #     """
    #     features = self.speech_featurizer.tf_extract(signal)
    #     encoded, new_encoder_states = self.encoder_inference(features, encoder_states)
    #     hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, prediction_states)
    #     transcript = self.text_featurizer.detokenize_unicode_points(hypothesis.prediction)
    #     return transcript, hypothesis.index, new_encoder_states, hypothesis.states

    # def recognize_tflite_with_timestamp(self, signal, predicted, encoder_states, prediction_states):
    #     features = self.speech_featurizer.tf_extract(signal)
    #     encoded, new_encoder_states = self.encoder_inference(features, encoder_states)
    #     hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, prediction_states)
    #     indices = self.text_featurizer.normalize_indices(hypothesis.prediction)
    #     upoints = tf.gather_nd(self.text_featurizer.upoints, tf.expand_dims(indices, axis=-1))  # [None, max_subword_length]

    #     num_samples = tf.cast(tf.shape(signal)[0], dtype=tf.float32)
    #     total_time_reduction_factor = self.time_reduction_factor * self.speech_featurizer.frame_step

    #     stime = tf.range(0, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
    #     stime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

    #     etime = tf.range(total_time_reduction_factor, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
    #     etime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

    #     non_blank = tf.where(tf.not_equal(upoints, 0))
    #     non_blank_transcript = tf.gather_nd(upoints, non_blank)
    #     non_blank_stime = tf.gather_nd(tf.repeat(tf.expand_dims(stime, axis=-1), tf.shape(upoints)[-1], axis=-1), non_blank)
    #     non_blank_etime = tf.gather_nd(tf.repeat(tf.expand_dims(etime, axis=-1), tf.shape(upoints)[-1], axis=-1), non_blank)

    #     return non_blank_transcript, non_blank_stime, non_blank_etime, hypothesis.index, new_encoder_states, hypothesis.states

    # -------------------------------- TFLITE -------------------------------------

    # def make_tflite_function(
    #     self,
    #     timestamp: bool = True,
    # ):
    #     tflite_func = self.recognize_tflite_with_timestamp if timestamp else self.recognize_tflite
    #     return tf.function(
    #         tflite_func,
    #         input_signature=[
    #             tf.TensorSpec([None], dtype=tf.float32),
    #             tf.TensorSpec([], dtype=tf.int32),
    #             tf.TensorSpec(self.encoder.get_initial_state().get_shape(), dtype=tf.float32),
    #             tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32),
    #         ],
    #     )
