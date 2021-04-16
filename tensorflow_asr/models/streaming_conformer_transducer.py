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
""" http://arxiv.org/abs/1811.06621 """

import tensorflow as tf

from .layers.subsampling import TimeReduction
# from .transducer import Transducer
from ..utils.utils import get_rnn, merge_two_last_dims, shape_list
from .conformer import Conformer

L2 = tf.keras.regularizers.l2(1e-6)

class StreamingConformer(Conformer):
    """
    Attempt at implementing Streaming Conformer Transducer. (see: https://arxiv.org/pdf/2010.11395.pdf).

    Three main differences:
    - Inputs are splits into chunks.
    - Masking is used for MHSA to select the chunks to be used at each timestep. (Allows for parallel training.)
    - Added parameter `streaming` to ConformerEncoder, ConformerBlock and ConvModule. Inside ConvModule, the layer DepthwiseConv2D has padding changed to "causal" when `streaming==True`.

    NOTE: Masking is applied just as regular masking along with the inputs.
    """
    def __init__(self,
                 vocabulary_size: int,
                 encoder_subsampling: dict,
                 encoder_positional_encoding: str = "sinusoid",
                 encoder_dmodel: int = 144,
                 encoder_num_blocks: int = 16,
                 encoder_head_size: int = 36,
                 encoder_num_heads: int = 4,
                 encoder_mha_type: str = "relmha",
                 encoder_kernel_size: int = 32,
                 encoder_depth_multiplier: int = 1,
                 encoder_fc_factor: float = 0.5,
                 encoder_dropout: float = 0,
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
                 name: str = "streaming_conformer",
                 **kwargs):

        self.streaming = True # Hardcoded value. Initializes Conformer with `streaming = True`.
        super(StreamingConformer, self).__init__(
            vocabulary_size=vocabulary_size,
            encoder_subsampling=encoder_subsampling,
            encoder_positional_encoding=encoder_positional_encoding,
            encoder_dmodel=encoder_dmodel,
            encoder_num_blocks=encoder_num_blocks,
            encoder_head_size=encoder_head_size,
            encoder_num_heads=encoder_num_heads,
            encoder_mha_type=encoder_mha_type,
            encoder_depth_multiplier=encoder_depth_multiplier,
            encoder_kernel_size=encoder_kernel_size,
            encoder_fc_factor=encoder_fc_factor,
            encoder_dropout=encoder_dropout,
            encoder_trainable=encoder_trainable,
            prediction_embed_dim=prediction_embed_dim,
            prediction_embed_dropout=prediction_embed_dropout,
            prediction_num_rnns=prediction_num_rnns,
            prediction_rnn_units=prediction_rnn_units,
            prediction_rnn_type=prediction_rnn_type,
            prediction_rnn_implementation=prediction_rnn_implementation,
            prediction_layer_norm=prediction_layer_norm,
            prediction_projection_units=prediction_projection_units,
            prediction_trainable=prediction_trainable,
            joint_dim=joint_dim,
            joint_activation=joint_activation,
            prejoint_linear=prejoint_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            joint_trainable=joint_trainable,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            streaming=self.streaming,
            name=name,
            **kwargs
        )
        self.dmodel = encoder_dmodel
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor

    def _build(self, input_shape, prediction_shape=[None], batch_size=None):
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        input_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        pred = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        pred_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        mask = tf.keras.Input(shape=[None, None], batch_size=batch_size, dtype=tf.int32)
        self([inputs, input_length, pred, pred_length, mask], training=False)

    def call(self, inputs, training=False, **kwargs):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            input_length: features time length in shape [B]
            prediction: predicted sequence of ids, in shape [B, U]
            prediction_length: predicted sequence of ids length in shape [B]
            mask: mask for streamed input frames [B, T, T]
            training: python boolean
            **kwargs: sth else

        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, _, prediction, prediction_length, mask = inputs
        enc = self.encoder(features, training=training, mask=mask, **kwargs) # Passes mask to encoder
        pred = self.predict_net([prediction, prediction_length], training=training, **kwargs)
        outputs = self.joint_net([enc, pred], training=training, **kwargs)
        return outputs

    def encoder_inference(self, features: tf.Tensor, states: tf.Tensor):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]
            states (tf.Tensor): previous states of encoders with shape [num_rnns, 1 or 2, 1, P]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
            tf.Tensor: states of encoders with shape [num_rnns, 1 or 2, 1, P]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs, new_states = self.encoder.recognize(outputs, states)
            return tf.squeeze(outputs, axis=0), new_states

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self,
                  features: tf.Tensor,
                  input_length: tf.Tensor,
                  parallel_iterations: int = 10,
                  swap_memory: bool = True):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of padded extracted features

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        batch_size, _, _, _ = shape_list(features)
        encoded, _ = self.encoder.recognize(features, self.encoder.get_initial_state(batch_size))
        return self._perform_greedy_batch(encoded, input_length,
                                          parallel_iterations=parallel_iterations, swap_memory=swap_memory)

    def recognize_tflite(self, signal, predicted, encoder_states, prediction_states):
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
            predicted: last predicted character with shape []
            encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
            prediction_states: lastest prediction states with shape [num_rnns, 1 or 2, 1, P]

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            predicted: last predicted character with shape []
            encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
            prediction_states: lastest prediction states with shape [num_rnns, 1 or 2, 1, P]
        """
        features = self.speech_featurizer.tf_extract(signal)
        encoded, new_encoder_states = self.encoder_inference(features, encoder_states)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, prediction_states)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, new_encoder_states, hypothesis.states

    def recognize_tflite_with_timestamp(self, signal, predicted, encoder_states, prediction_states):
        features = self.speech_featurizer.tf_extract(signal)
        encoded, new_encoder_states = self.encoder_inference(features, encoder_states)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, prediction_states)
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

        return non_blank_transcript, non_blank_stime, non_blank_etime, hypothesis.index, new_encoder_states, hypothesis.states

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self,
                       features: tf.Tensor,
                       input_length: tf.Tensor,
                       lm: bool = False,
                       parallel_iterations: int = 10,
                       swap_memory: bool = True):
        """
        RNN Transducer Beam Search
        Args:
            features (tf.Tensor): a batch of padded extracted features
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        batch_size, _, _, _ = shape_list(features)
        encoded, _ = self.encoder.recognize(features, self.encoder.get_initial_state(batch_size))
        return self._perform_beam_search_batch(encoded, input_length, lm,
                                               parallel_iterations=parallel_iterations, swap_memory=swap_memory)

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(self, timestamp: bool = True):
        tflite_func = self.recognize_tflite_with_timestamp if timestamp else self.recognize_tflite
        return tf.function(
            tflite_func,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.encoder.get_initial_state().get_shape(), dtype=tf.float32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32)
            ]
        )
