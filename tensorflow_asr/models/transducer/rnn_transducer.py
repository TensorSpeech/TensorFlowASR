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

from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.layers.subsampling import TimeReduction
from tensorflow_asr.models.transducer.base_transducer import Transducer
from tensorflow_asr.utils import layer_util, math_util


class Reshape(Layer):
    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = math_util.merge_two_last_dims(outputs)
        outputs = math_util.apply_mask(outputs, mask=tf.sequence_mask(outputs_length, maxlen=tf.shape(outputs)[1], dtype=tf.bool))
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = list(output_shape)
        return (output_shape[0], output_shape[1], output_shape[2] * output_shape[3]), tuple(output_length_shape)


class RnnTransducerBlock(Layer):
    def __init__(
        self,
        reduction_factor: int = 0,
        dmodel: int = 640,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        rnn_unroll: bool = False,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        RnnClass = layer_util.get_rnn(rnn_type)
        self.rnn = RnnClass(
            units=rnn_units,
            return_sequences=True,
            name=rnn_type,
            unroll=rnn_unroll,
            return_state=True,
            zero_output_for_mask=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        if layer_norm:
            self.ln = tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
        else:
            self.ln = None

        if reduction_factor > 0:
            self.reduction = TimeReduction(reduction_factor, name="reduction")
        else:
            self.reduction = None

        self.projection = tf.keras.layers.Dense(dmodel, name="projection", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.rnn(outputs, training=training, mask=getattr(outputs, "_keras_mask", None))
        outputs = outputs[0]
        if self.ln is not None:
            outputs = self.ln(outputs, training=training)
        if self.reduction is not None:
            outputs, outputs_length = self.reduction([outputs, outputs_length])
        outputs = self.projection(outputs, training=training)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        if self.reduction is not None:
            mask = self.reduction.compute_mask(inputs)
        return mask

    def recognize(self, inputs, states):
        outputs = inputs
        outputs = self.rnn(outputs, training=False, initial_state=states, mask=getattr(outputs, "_keras_mask", None))
        new_states = tf.stack(outputs[1:], axis=0)
        outputs = outputs[0]
        if self.ln is not None:
            outputs = self.ln(outputs, training=False)
        if self.reduction is not None:
            outputs, _ = self.reduction([outputs, tf.reshape(tf.shape(outputs)[1], [1])])
        outputs = self.projection(outputs, training=False)
        return outputs, new_states

    def compute_output_shape(self, input_shape):
        if self.reduction is None:
            return tuple(input_shape)
        return self.reduction.compute_output_shape(input_shape)


class RnnTransducerEncoder(Layer):
    def __init__(
        self,
        reductions: dict = {0: 3, 1: 2},
        dmodel: int = 640,
        nlayers: int = 8,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        rnn_unroll: bool = False,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dmodel = dmodel
        self.reshape = Reshape(name="reshape")

        self.blocks = [
            RnnTransducerBlock(
                reduction_factor=reductions.get(i, 0) if reductions else 0,  # key is index, value is the factor
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                rnn_unroll=rnn_unroll,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            for i in range(nlayers)
        ]

        self.time_reduction_factor = 1
        for block in self.blocks:
            if block.reduction is not None:
                self.time_reduction_factor *= block.reduction.time_reduction_factor

    def get_initial_state(self, batch_size=1):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, P]
        """
        states = []
        for block in self.blocks:
            states.append(tf.stack(block.rnn.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=tf.float32)), axis=0))
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False):
        outputs, outputs_length = self.reshape(inputs)
        for block in self.blocks:
            outputs, outputs_length = block([outputs, outputs_length], training=training)
        return outputs, outputs_length

    def recognize(self, inputs, states):
        """Recognize function for encoder network

        Args:
            inputs (tf.Tensor): shape [1, T, F, C]
            states (tf.Tensor): shape [num_lstms, 1 or 2, 1, P]

        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [num_lstms, 1 or 2, 1, P]
        """
        outputs, _ = self.reshape([inputs, tf.reshape(tf.shape(inputs)[1], [1])])
        new_states = []
        for i, block in enumerate(self.blocks):
            outputs, block_states = block.recognize(outputs, states=tf.unstack(states[i], axis=0))
            new_states.append(block_states)
        return outputs, tf.stack(new_states, axis=0)

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = self.reshape.compute_output_shape(input_shape)
        output_shape = list(output_shape)
        output_shape[1] = None if output_shape[1] is None else math_util.legacy_get_reduced_length(output_shape[1], self.time_reduction_factor)
        output_shape[2] = self._dmodel
        return tuple(output_shape), output_length_shape


class RnnTransducer(Transducer):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        encoder_reductions: dict = {0: 3, 1: 2},
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
            encoder=RnnTransducerEncoder(
                reductions=encoder_reductions,
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

    def encoder_inference(self, features: tf.Tensor, states: tf.Tensor):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]
            states (tf.Tensor): previous states of encoders with shape [num_rnns, 1 or 2, 1, P]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
            tf.Tensor: states of encoders with shape [num_rnns, 1 or 2, 1, P]
        """
        with tf.name_scope("encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs, new_states = self.encoder.recognize(outputs, states)
            return tf.squeeze(outputs, axis=0), new_states

    # -------------------------------- GREEDY -------------------------------------

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

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(
        self,
        timestamp: bool = True,
    ):
        tflite_func = self.recognize_tflite_with_timestamp if timestamp else self.recognize_tflite
        return tf.function(
            tflite_func,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.encoder.get_initial_state().get_shape(), dtype=tf.float32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32),
            ],
        )
