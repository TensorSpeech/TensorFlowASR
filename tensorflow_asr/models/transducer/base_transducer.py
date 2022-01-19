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
""" https://arxiv.org/pdf/1811.06621.pdf """

import collections
from typing import Dict, Tuple
import tensorflow as tf

from tensorflow_asr.models.transducer.transducer_prediction import TransducerPrediction
from tensorflow_asr.models.transducer.transducer_joint import TransducerJoint
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.losses.rnnt_loss import RnntLoss
from tensorflow_asr.mwer.beam_search import BeamSearch
from tensorflow_asr.mwer.mwer_loss import MWERLoss
from tensorflow_asr.mwer.wer import WER
from tensorflow_asr.utils import data_util, layer_util, math_util, shape_util
from tensorflow_asr.models.base_model import BaseModel

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))


class Transducer(BaseModel):
    """Transducer Model Warper"""

    def __init__(
            self,
            encoder: tf.keras.Model,
            vocabulary_size: int,
            blank_token: int = 0,  # TODO: clean up the code such that blank is non-optional argument
            # beam_size: int = 2, # TODO: change to 1
            embed_dim: int = 512,
            embed_dropout: float = 0,
            num_rnns: int = 1,
            rnn_units: int = 320,
            rnn_type: str = "lstm",
            rnn_implementation: int = 2,
            layer_norm: bool = True,
            projection_units: int = 0,
            prediction_trainable: bool = True,
            joint_dim: int = 1024,
            joint_activation: str = "tanh",
            prejoint_linear: bool = True,
            postjoint_linear: bool = False,
            joint_mode: str = "add",
            joint_trainable: bool = True,
            kernel_regularizer=None,
            bias_regularizer=None,
            mwer_training=False,
            name="transducer",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_rnns=num_rnns,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            rnn_implementation=rnn_implementation,
            layer_norm=layer_norm,
            projection_units=projection_units,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=prediction_trainable,
            name=f"{name}_prediction",
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            activation=joint_activation,
            prejoint_linear=prejoint_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=joint_trainable,
            name=f"{name}_joint",
        )
        self.beam = BeamSearch(
            vocabulary_size=vocabulary_size,
            predict_net=self.predict_net,
            joint_net=self.joint_net,
            blank_token=blank_token,
            name=f"{name}_beam_search"
        )
        self.time_reduction_factor = 1
        self._beam_size = None
        self._batch_size = None
        self._mwer_training = mwer_training

    def make(
            self,
            input_shape,
            prediction_shape=[None],
            batch_size=None,
    ):
        self._batch_size = batch_size
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length,
                predictions=predictions,
                predictions_length=predictions_length,
            ),
            training=False,
        )

    def summary(
            self,
            line_length=None,
            **kwargs,
    ):
        if self.encoder is not None:
            self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(
            self,
            speech_featurizer: SpeechFeaturizer,
            text_featurizer: TextFeaturizer,
    ):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.beam.beam_size = text_featurizer.decoder_config.beam_width
        self._beam_size = text_featurizer.decoder_config.beam_width

    def compile(
            self,
            optimizer,
            global_batch_size,
            blank=0,
            run_eagerly=None,
            **kwargs,
    ):
        if self._mwer_training:
            loss = MWERLoss(risk_obj=WER(), blank=blank, global_batch_size=global_batch_size)
        else:
            loss = RnntLoss(blank=blank, global_batch_size=global_batch_size)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def call(
            self,
            inputs,
            training=False,
            **kwargs,
    ):
        enc = self.encoder(inputs["inputs"], training=training, **kwargs)
        pred = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=training, **kwargs)
        logits = self.joint_net([enc, pred], training=training, **kwargs)

        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor),
        )

    def train_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of training data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric

        """
        if not self._mwer_training:
            return super().train_step(batch)

        inputs, y_true = batch
        features, features_lengths = self._copy_input_features(inputs)
        sentences, sentences_lengths, hypotheses = self._get_beam_hypotheses(inputs, y_true)

        blank_slice = tf.ones([tf.shape(sentences)[0], 1], dtype=tf.int32) * self.text_featurizer.blank
        rnnt_input_sentences = tf.concat([blank_slice, sentences], axis=1)
        rnnt_inputs = data_util.create_inputs(features,
                                              features_lengths,
                                              rnnt_input_sentences,
                                              sentences_lengths + tf.constant(1))
        ground_truths = data_util.create_labels(sentences, sentences_lengths)

        with tf.GradientTape() as tape:
            y_pred = self(rnnt_inputs, training=True)
            loss = self.loss(y_pred, ground_truths, hypotheses)
            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_loss_scale:
            gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._tfasr_metrics["loss"].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]: a batch of validation data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric prefixed with "val_"

        """
        if not self._mwer_training:
            return super().test_step(batch)

        inputs, y_true = batch
        features, features_lengths = self._copy_input_features(inputs)
        sentences, sentences_lengths, hypotheses = self._get_beam_hypotheses(inputs, y_true)

        blank_slice = tf.ones([tf.shape(sentences)[0], 1], dtype=tf.int32) * self.text_featurizer.blank
        rnnt_input_sentences = tf.concat([blank_slice, sentences], axis=1)
        rnnt_inputs = data_util.create_inputs(features,
                                              features_lengths,
                                              rnnt_input_sentences,
                                              sentences_lengths + tf.constant(1))
        ground_truths = data_util.create_labels(sentences, sentences_lengths)
        y_pred = self(rnnt_inputs, training=True)
        loss = self.loss(y_pred, ground_truths, hypotheses)
        self._tfasr_metrics["loss"].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def _copy_input_features(self,
                             inputs: Dict[str, tf.Tensor],
                             ) -> Tuple[tf.Tensor, tf.Tensor]:
        features = inputs["inputs"]
        features_length = inputs["inputs_length"]

        # copying input features lengths for each prediction
        features_length = tf.expand_dims(features_length, axis=0)
        features_length = tf.transpose(features_length)
        features_length = tf.tile(features_length, [1, self._beam_size])
        features_length = tf.reshape(features_length, [-1])

        # copying input features for each prediction
        features_shape = tf.shape(features)
        features = tf.tile(features, [self._beam_size, 1, 1, 1])
        features = tf.reshape(features, [self._beam_size] + tf.unstack(features_shape))
        features = tf.transpose(features, [1, 0, 2, 3, 4])
        features = tf.reshape(features, [-1] + tf.unstack(features_shape[1:]))

        return features, features_length

    def _get_beam_hypotheses(self,
                             inputs: Dict[str, tf.Tensor],
                             y_true: Dict[str, tf.Tensor]
                             ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        labels_transcriptions = self.text_featurizer.iextract(y_true["labels"])
        transcriptions, sentences, log_probas = self.recognize_beam(inputs,
                                                                    return_tokens=True,
                                                                    return_topk=True,
                                                                    return_log_probas=True)
        transcriptions = tf.reshape(transcriptions, [-1])
        log_probas = tf.reshape(log_probas, [-1])
        sentences = tf.reshape(sentences, [-1, tf.shape(sentences)[2]])
        labels_transcriptions = tf.expand_dims(labels_transcriptions, axis=1)
        labels_transcriptions = tf.tile(labels_transcriptions, [1, self._beam_size])
        labels_transcriptions = tf.reshape(labels_transcriptions, [-1])

        hypotheses = data_util.create_hypotheses(sentences=transcriptions,
                                                 log_probas=log_probas,
                                                 labels=labels_transcriptions)

        nonblank_tokens = tf.math.not_equal(sentences, self.text_featurizer.blank)
        nonblank_tokens = tf.cast(nonblank_tokens, dtype=tf.int32)
        sentences_length = tf.reduce_sum(nonblank_tokens, axis=1)
        max_length = tf.reduce_max(sentences_length)

        def remove_zeros_and_pad(input: tf.Tensor):
            nonblank = tf.where(input != tf.constant(self.text_featurizer.blank))
            nonblank_count = tf.math.count_nonzero(nonblank, dtype=tf.int32)
            nonblank_tokens = tf.gather_nd(input, nonblank)

            return tf.pad(nonblank_tokens, [[0, max_length - nonblank_count]])

        sentences = tf.map_fn(remove_zeros_and_pad, sentences)

        return sentences, sentences_length, hypotheses
    # -------------------------------- INFERENCES -------------------------------------

    def encoder_inference(
            self,
            features: tf.Tensor,
    ):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs = self.encoder(outputs, training=False)
            return tf.squeeze(outputs, axis=0)

    def decoder_inference(
            self,
            encoded: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            tflite: bool = False,
    ):
        """Infer function for decoder

        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predict_net.recognize(predicted, states, tflite=tflite)  # [1, 1, P], states
            ytu = tf.nn.log_softmax(self.joint_net([encoded, y], training=False))  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(
            self,
            inputs: Dict[str, tf.Tensor],
    ):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of extracted features
            input_length (tf.Tensor): a batch of extracted features length

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_greedy_batch(encoded=encoded, encoded_length=encoded_length)

    def recognize_tflite(
            self,
            signal,
            predicted,
            states,
    ):
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
            predicted: last predicted character with shape []
            states: lastest rnn states with shape [num_rnns, 1 or 2, 1, P]

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            predicted: last predicted character with shape []
            states: lastest rnn states with shape [num_rnns, 1 or 2, 1, P]
        """
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)

        return transcript, hypothesis.index, hypothesis.states

    def recognize_tflite_with_timestamp(
            self,
            signal,
            predicted,
            states,
    ):
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
        indices = self.text_featurizer.normalize_indices(hypothesis.prediction)
        upoints = tf.gather_nd(self.text_featurizer.upoints,
                               tf.expand_dims(indices, axis=-1))  # [None, max_subword_length]

        num_samples = tf.cast(tf.shape(signal)[0], dtype=tf.float32)
        total_time_reduction_factor = self.time_reduction_factor * self.speech_featurizer.frame_step

        stime = tf.range(0, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
        stime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

        etime = tf.range(total_time_reduction_factor, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
        etime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

        non_blank = tf.where(tf.not_equal(upoints, 0))
        non_blank_transcript = tf.gather_nd(upoints, non_blank)
        non_blank_stime = tf.gather_nd(tf.repeat(tf.expand_dims(stime, axis=-1), tf.shape(upoints)[-1], axis=-1),
                                       non_blank)
        non_blank_etime = tf.gather_nd(tf.repeat(tf.expand_dims(etime, axis=-1), tf.shape(upoints)[-1], axis=-1),
                                       non_blank)

        return non_blank_transcript, non_blank_stime, non_blank_etime, hypothesis.index, hypothesis.states

    def _perform_greedy_batch(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
    ):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([None]),
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_greedy(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    states=self.predict_net.get_initial_state(),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())

    def _perform_greedy(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
            tflite: bool = False,
    ):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=total,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []
                # something is wrong with tflite that drop support for tf.cond
                # def equal_blank_fn(): return _hypothesis.index, _hypothesis.states
                # def non_equal_blank_fn(): return _predict, _states  # update if the new prediction is a non-blank
                # _index, _states = tf.cond(tf.equal(_predict, blank), equal_blank_fn, non_equal_blank_fn)

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)
                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time + 1, _hypothesis

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )
            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )

    def _perform_greedy_v2(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
            tflite: bool = False,
    ):
        """Ref: https://arxiv.org/pdf/1801.00841.pdf"""
        with tf.name_scope(f"{self.name}_greedy_v2"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)
                _time = tf.where(_equal, _time + 1, _time)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time, _hypothesis

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(
            self,
            inputs: Dict[str, tf.Tensor],
            return_tokens: bool = False,
            return_topk: bool = False,
            return_log_probas: bool = False,
            lm: bool = False,
    ):
        """
        RNN Transducer Beam Search
        Args:
            inputs (Dict[str, tf.Tensor]): Input dictionary containing "inputs" and "inputs_length"
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        predictions, probabilities = self.beam.call(encoded=encoded,
                                                    encoded_length=encoded_length,
                                                    return_topk=return_topk,
                                                    parallel_iterations=256)
        transcriptions = tf.map_fn(self.text_featurizer.iextract, predictions, fn_output_signature=tf.string)
        output = [transcriptions]
        if return_tokens:
            output = output + [predictions]
        if return_log_probas:
            output = output + [probabilities]

        return output

    def _perform_beam_search_batch(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            lm: bool = False,
            parallel_iterations: int = 10,
            swap_memory: bool = True,
    ):
        with tf.name_scope(f"{self.name}_perform_beam_search_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=None,
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_beam_search(
                    encoded[batch],
                    encoded_length[batch],
                    lm,
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())

    def _perform_beam_search(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            lm: bool = False,
            parallel_iterations: int = 10,
            swap_memory: bool = True,
            tflite: bool = False,
    ):
        with tf.name_scope(f"{self.name}_beam_search"):
            beam_width = tf.cond(
                tf.less(self.text_featurizer.decoder_config.beam_width, self.text_featurizer.num_classes),
                true_fn=lambda: self.text_featurizer.decoder_config.beam_width,
                false_fn=lambda: self.text_featurizer.num_classes - 1,
            )
            total = encoded_length

            def initialize_beam(dynamic=False):
                return BeamHypothesis(
                    score=tf.TensorArray(
                        dtype=tf.float32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]),
                        clear_after_read=False,
                    ),
                    indices=tf.TensorArray(
                        dtype=tf.int32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]),
                        clear_after_read=False,
                    ),
                    prediction=tf.TensorArray(
                        dtype=tf.int32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=None,
                        clear_after_read=False,
                    ),
                    states=tf.TensorArray(
                        dtype=tf.float32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape(shape_util.shape_list(self.predict_net.get_initial_state())),
                        clear_after_read=False,
                    ),
                )

            B = initialize_beam()
            B = BeamHypothesis(
                score=B.score.write(0, 0.0),
                indices=B.indices.write(0, self.text_featurizer.blank),
                prediction=B.prediction.write(0, tf.ones([total], dtype=tf.int32) * self.text_featurizer.blank),
                states=B.states.write(0, self.predict_net.get_initial_state()),
            )

            def condition(time, total, B):
                return tf.less(time, total)

            def body(time, total, B):
                A = initialize_beam(dynamic=True)
                A = BeamHypothesis(
                    score=A.score.unstack(B.score.stack()),
                    indices=A.indices.unstack(B.indices.stack()),
                    prediction=A.prediction.unstack(
                        math_util.pad_prediction_tfarray(B.prediction, blank=self.text_featurizer.blank).stack()
                    ),
                    states=A.states.unstack(B.states.stack()),
                )
                A_i = tf.constant(0, tf.int32)
                B = initialize_beam()

                encoded_t = tf.gather_nd(encoded, tf.expand_dims(time, axis=-1))

                def beam_condition(beam, beam_width, A, A_i, B):
                    return tf.less(beam, beam_width)

                def beam_body(beam, beam_width, A, A_i, B):
                    # get y_hat
                    y_hat_score, y_hat_score_index = tf.math.top_k(A.score.stack(), k=1, sorted=True)
                    y_hat_score = y_hat_score[0]
                    y_hat_index = tf.gather_nd(A.indices.stack(), y_hat_score_index)
                    y_hat_prediction = tf.gather_nd(
                        math_util.pad_prediction_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
                        y_hat_score_index,
                    )
                    y_hat_states = tf.gather_nd(A.states.stack(), y_hat_score_index)

                    # remove y_hat from A
                    remain_indices = tf.range(0, tf.shape(A.score.stack())[0], dtype=tf.int32)
                    remain_indices = tf.gather_nd(remain_indices,
                                                  tf.where(tf.not_equal(remain_indices, y_hat_score_index[0])))
                    remain_indices = tf.expand_dims(remain_indices, axis=-1)
                    A = BeamHypothesis(
                        score=A.score.unstack(tf.gather_nd(A.score.stack(), remain_indices)),
                        indices=A.indices.unstack(tf.gather_nd(A.indices.stack(), remain_indices)),
                        prediction=A.prediction.unstack(
                            tf.gather_nd(
                                math_util.pad_prediction_tfarray(A.prediction,
                                                                 blank=self.text_featurizer.blank).stack(),
                                remain_indices,
                            )
                        ),
                        states=A.states.unstack(tf.gather_nd(A.states.stack(), remain_indices)),
                    )
                    A_i = tf.cond(tf.equal(A_i, 0), true_fn=lambda: A_i, false_fn=lambda: A_i - 1)

                    ytu, new_states = self.decoder_inference(
                        encoded=encoded_t, predicted=y_hat_index, states=y_hat_states, tflite=tflite
                    )

                    def predict_condition(pred, A, A_i, B):
                        return tf.less(pred, self.text_featurizer.num_classes)

                    def predict_body(pred, A, A_i, B):
                        new_score = y_hat_score + tf.gather_nd(ytu, tf.expand_dims(pred, axis=-1))

                        def true_fn():
                            return (
                                B.score.write(beam, new_score),
                                B.indices.write(beam, y_hat_index),
                                B.prediction.write(beam, y_hat_prediction),
                                B.states.write(beam, y_hat_states),
                                A.score,
                                A.indices,
                                A.prediction,
                                A.states,
                                A_i,
                            )

                        def false_fn():
                            scatter_index = math_util.count_non_blank(y_hat_prediction,
                                                                      blank=self.text_featurizer.blank)
                            updated_prediction = tf.tensor_scatter_nd_update(
                                y_hat_prediction,
                                indices=tf.reshape(scatter_index, [1, 1]),
                                updates=tf.expand_dims(pred, axis=-1),
                            )
                            return (
                                B.score,
                                B.indices,
                                B.prediction,
                                B.states,
                                A.score.write(A_i, new_score),
                                A.indices.write(A_i, pred),
                                A.prediction.write(A_i, updated_prediction),
                                A.states.write(A_i, new_states),
                                A_i + 1,
                            )

                        b_score, b_indices, b_prediction, b_states, a_score, a_indices, a_prediction, a_states, A_i = tf.cond(
                            tf.equal(pred, self.text_featurizer.blank), true_fn=true_fn, false_fn=false_fn
                        )

                        B = BeamHypothesis(score=b_score, indices=b_indices, prediction=b_prediction, states=b_states)
                        A = BeamHypothesis(score=a_score, indices=a_indices, prediction=a_prediction, states=a_states)

                        return pred + 1, A, A_i, B

                    _, A, A_i, B = tf.while_loop(
                        predict_condition,
                        predict_body,
                        loop_vars=[0, A, A_i, B],
                        parallel_iterations=parallel_iterations,
                        swap_memory=swap_memory,
                    )

                    return beam + 1, beam_width, A, A_i, B

                _, _, A, A_i, B = tf.while_loop(
                    beam_condition,
                    beam_body,
                    loop_vars=[0, beam_width, A, A_i, B],
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )

                return time + 1, total, B

            _, _, B = tf.while_loop(
                condition,
                body,
                loop_vars=[0, total, B],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            scores = B.score.stack()
            prediction = math_util.pad_prediction_tfarray(B.prediction, blank=self.text_featurizer.blank).stack()
            if self.text_featurizer.decoder_config.norm_score:
                prediction_lengths = math_util.count_non_blank(prediction, blank=self.text_featurizer.blank, axis=1)
                scores /= tf.cast(prediction_lengths, dtype=scores.dtype)

            y_hat_score, y_hat_score_index = tf.math.top_k(scores, k=1)
            y_hat_score = y_hat_score[0]
            y_hat_index = tf.gather_nd(B.indices.stack(), y_hat_score_index)
            y_hat_prediction = tf.gather_nd(prediction, y_hat_score_index)
            y_hat_states = tf.gather_nd(B.states.stack(), y_hat_score_index)

            return Hypothesis(index=y_hat_index, prediction=y_hat_prediction, states=y_hat_states)

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(
            self,
            timestamp: bool = False,
    ):
        tflite_func = self.recognize_tflite_with_timestamp if timestamp else self.recognize_tflite
        return tf.function(
            tflite_func,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32),
            ],
        )
