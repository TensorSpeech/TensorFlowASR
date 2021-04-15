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
from typing import Dict
import tensorflow as tf

from ..base_model import BaseModel
from ...utils import math_util, layer_util, shape_util, data_util
from ...featurizers.speech_featurizers import SpeechFeaturizer
from ...featurizers.text_featurizers import TextFeaturizer
from ..layers.embedding import Embedding
from ...losses.rnnt_loss import RnntLoss

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))


class TransducerPrediction(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 512,
                 rnn_type: str = "lstm",
                 rnn_implementation: int = 2,
                 layer_norm: bool = True,
                 projection_units: int = 0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="transducer_prediction",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed = Embedding(vocabulary_size, embed_dim,
                               regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        # Initialize rnn layers
        RNN = layer_util.get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units=rnn_units, return_sequences=True,
                name=f"{name}_{rnn_type}_{i}", return_state=True,
                implementation=rnn_implementation,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            if layer_norm:
                ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}")
            else:
                ln = None
            if projection_units > 0:
                projection = tf.keras.layers.Dense(
                    projection_units,
                    name=f"{name}_projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer
                )
            else:
                projection = None
            self.rnns.append({"rnn": rnn, "ln": ln, "projection": projection})

    def get_initial_state(self):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(
                tf.stack(
                    rnn["rnn"].get_initial_state(
                        tf.zeros([1, 1, 1], dtype=tf.float32)
                    ), axis=0
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False, **kwargs):
        # inputs has shape [B, U]
        # use tf.gather_nd instead of tf.gather for tflite conversion
        outputs, prediction_length = inputs
        outputs = self.embed(outputs, training=training)
        outputs = self.do(outputs, training=training)
        for rnn in self.rnns:
            mask = tf.sequence_mask(prediction_length, maxlen=tf.shape(outputs)[1])
            outputs = rnn["rnn"](outputs, training=training, mask=mask)
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=training)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for prediction network

        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]

        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, training=False, initial_state=tf.unstack(states[i], axis=0))
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.embed.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn["rnn"].get_config())
            if rnn["ln"] is not None:
                conf.update(rnn["ln"].get_config())
            if rnn["projection"] is not None:
                conf.update(rnn["projection"].get_config())
        return conf


class TransducerJointReshape(tf.keras.layers.Layer):
    def __init__(self,
                 axis: int = 1,
                 name="transducer_joint_reshape",
                 **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.axis = axis

    def call(self, inputs, repeats=None, **kwargs):
        outputs = tf.expand_dims(inputs, axis=self.axis)
        return tf.repeat(outputs, repeats=repeats, axis=self.axis)

    def get_config(self):
        conf = super(TransducerJointReshape, self).get_config()
        conf.update({"axis": self.axis})
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 activation: str = "tanh",
                 prejoint_linear: bool = True,
                 postjoint_linear: bool = False,
                 joint_mode: str = "add",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="tranducer_joint",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        activation = activation.lower()
        if activation == "linear":
            self.activation = tf.keras.layers.Activation(tf.keras.activation.linear, name=f"{name}_linear")
        elif activation == "relu":
            self.activation = tf.keras.layers.Activation(tf.nn.relu, name=f"{name}_relu")
        elif activation == "tanh":
            self.activation = tf.keras.layers.Activation(tf.nn.tanh, name=f"{name}_tanh")
        else:
            raise ValueError("activation must be either 'linear', 'relu' or 'tanh'")

        self.prejoint_linear = prejoint_linear
        self.postjoint_linear = postjoint_linear

        if self.prejoint_linear:
            self.ffn_enc = tf.keras.layers.Dense(
                joint_dim, name=f"{name}_enc",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            self.ffn_pred = tf.keras.layers.Dense(
                joint_dim, use_bias=False, name=f"{name}_pred",
                kernel_regularizer=kernel_regularizer
            )

        self.enc_reshape = TransducerJointReshape(axis=2, name=f"{name}_enc_reshape")
        self.pred_reshape = TransducerJointReshape(axis=1, name=f"{name}_pred_reshape")

        if joint_mode == "add":
            self.joint = tf.keras.layers.Add(name=f"{name}_add")
        elif joint_mode == "concat":
            self.joint = tf.keras.layers.Concatenate(name=f"{name}_concat")
        else:
            raise ValueError("joint_mode must be either 'add' or 'concat'")

        if self.postjoint_linear:
            self.ffn = tf.keras.layers.Dense(
                joint_dim, name=f"{name}_ffn",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )

        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size, name=f"{name}_vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        if self.prejoint_linear:
            enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
            pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        enc_out = self.enc_reshape(enc_out, repeats=tf.shape(pred_out)[1])
        pred_out = self.pred_reshape(pred_out, repeats=tf.shape(enc_out)[1])
        outputs = self.joint([enc_out, pred_out], training=training)
        if self.postjoint_linear:
            outputs = self.ffn(outputs, training=training)
        outputs = self.activation(outputs, training=training)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        conf.update(self.activation.get_config())
        conf.update(self.joint.get_config())
        conf.update({"prejoint_linear": self.prejoint_linear, "postjoint_linear": self.postjoint_linear})
        return conf


class Transducer(BaseModel):
    """ Transducer Model Warper """

    def __init__(self,
                 encoder: tf.keras.Model,
                 vocabulary_size: int,
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
                 name="transducer",
                 **kwargs):
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
            name=f"{name}_prediction"
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
            name=f"{name}_joint"
        )
        self.time_reduction_factor = 1

    def _build(self, input_shape, prediction_shape=[None], batch_size=None):
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length,
                predictions=predictions,
                predictions_length=predictions_length
            ),
            training=False
        )

    def summary(self, line_length=None, **kwargs):
        if self.encoder is not None: self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: SpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def compile(self,
                optimizer,
                global_batch_size,
                blank=0,
                run_eagerly=None,
                **kwargs):
        loss = RnntLoss(blank=blank, global_batch_size=global_batch_size)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs, training=False, **kwargs):
        enc = self.encoder(inputs["inputs"], training=training, **kwargs)
        pred = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=training, **kwargs)
        logits = self.joint_net([enc, pred], training=training, **kwargs)
        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        )

    # -------------------------------- INFERENCES -------------------------------------

    def encoder_inference(self, features: tf.Tensor):
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

    def decoder_inference(self, encoded: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor):
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
            y, new_states = self.predict_net.recognize(predicted, states)  # [1, 1, P], states
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
    def recognize(self, inputs: Dict[str, tf.Tensor]):
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

    def recognize_tflite(self, signal, predicted, states):
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
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, hypothesis.states

    def recognize_tflite_with_timestamp(self, signal, predicted, states):
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features)
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

    def _perform_greedy_batch(self,
                              encoded: tf.Tensor,
                              encoded_length: tf.Tensor,
                              parallel_iterations: int = 10,
                              swap_memory: bool = False):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32, size=total_batch, dynamic_size=False,
                clear_after_read=False, element_shape=tf.TensorShape([None])
            )

            def condition(batch, _): return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_greedy(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    states=self.predict_net.get_initial_state(),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition, body, loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations, swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())

    def _perform_greedy(self,
                        encoded: tf.Tensor,
                        encoded_length: tf.Tensor,
                        predicted: tf.Tensor,
                        states: tf.Tensor,
                        parallel_iterations: int = 10,
                        swap_memory: bool = False):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32, size=total, dynamic_size=False,
                    clear_after_read=False, element_shape=tf.TensorShape([])
                ),
                states=states
            )

            def condition(_time, _hypothesis): return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states
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
                condition, body,
                loop_vars=[time, hypothesis], parallel_iterations=parallel_iterations, swap_memory=swap_memory
            )

            return Hypothesis(index=hypothesis.index, prediction=hypothesis.prediction.stack(), states=hypothesis.states)

    def _perform_greedy_v2(self,
                           encoded: tf.Tensor,
                           encoded_length: tf.Tensor,
                           predicted: tf.Tensor,
                           states: tf.Tensor,
                           parallel_iterations: int = 10,
                           swap_memory: bool = False):
        """ Ref: https://arxiv.org/pdf/1801.00841.pdf """
        with tf.name_scope(f"{self.name}_greedy_v2"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32, size=0, dynamic_size=True,
                    clear_after_read=False, element_shape=tf.TensorShape([])
                ),
                states=states
            )

            def condition(_time, _hypothesis): return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states
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
                condition, body,
                loop_vars=[time, hypothesis], parallel_iterations=parallel_iterations, swap_memory=swap_memory
            )

            return Hypothesis(index=hypothesis.index, prediction=hypothesis.prediction.stack(), states=hypothesis.states)

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
        encoded = self.encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_beam_search_batch(encoded=encoded, encoded_length=encoded_length, lm=lm)

    def _perform_beam_search_batch(self,
                                   encoded: tf.Tensor,
                                   encoded_length: tf.Tensor,
                                   lm: bool = False,
                                   parallel_iterations: int = 10,
                                   swap_memory: bool = False):
        with tf.name_scope(f"{self.name}_perform_beam_search_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32, size=total_batch, dynamic_size=False,
                clear_after_read=False, element_shape=None
            )

            def condition(batch, _): return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_beam_search(
                    encoded[batch], encoded_length[batch], lm,
                    parallel_iterations=parallel_iterations, swap_memory=swap_memory
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition, body, loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations, swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())

    def _perform_beam_search(self,
                             encoded: tf.Tensor,
                             encoded_length: tf.Tensor,
                             lm: bool = False,
                             parallel_iterations: int = 10,
                             swap_memory: bool = False):
        with tf.name_scope(f"{self.name}_beam_search"):
            beam_width = tf.cond(
                tf.less(self.text_featurizer.decoder_config.beam_width, self.text_featurizer.num_classes),
                true_fn=lambda: self.text_featurizer.decoder_config.beam_width,
                false_fn=lambda: self.text_featurizer.num_classes - 1
            )
            total = encoded_length

            def initialize_beam(dynamic=False):
                return BeamHypothesis(
                    score=tf.TensorArray(
                        dtype=tf.float32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]),
                        clear_after_read=False
                    ),
                    indices=tf.TensorArray(
                        dtype=tf.int32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]),
                        clear_after_read=False
                    ),
                    prediction=tf.TensorArray(
                        dtype=tf.int32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=None,
                        clear_after_read=False
                    ),
                    states=tf.TensorArray(
                        dtype=tf.float32,
                        size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape(shape_util.shape_list(self.predict_net.get_initial_state())),
                        clear_after_read=False
                    ),
                )

            B = initialize_beam()
            B = BeamHypothesis(
                score=B.score.write(0, 0.0),
                indices=B.indices.write(0, self.text_featurizer.blank),
                prediction=B.prediction.write(0, tf.ones([total], dtype=tf.int32) * self.text_featurizer.blank),
                states=B.states.write(0, self.predict_net.get_initial_state())
            )

            def condition(time, total, B): return tf.less(time, total)

            def body(time, total, B):
                A = initialize_beam(dynamic=True)
                A = BeamHypothesis(
                    score=A.score.unstack(B.score.stack()),
                    indices=A.indices.unstack(B.indices.stack()),
                    prediction=A.prediction.unstack(
                        math_util.pad_prediction_tfarray(
                            B.prediction,
                            blank=self.text_featurizer.blank
                        ).stack()
                    ),
                    states=A.states.unstack(B.states.stack()),
                )
                A_i = tf.constant(0, tf.int32)
                B = initialize_beam()

                encoded_t = tf.gather_nd(encoded, tf.expand_dims(time, axis=-1))

                def beam_condition(beam, beam_width, A, A_i, B): return tf.less(beam, beam_width)

                def beam_body(beam, beam_width, A, A_i, B):
                    # get y_hat
                    y_hat_score, y_hat_score_index = tf.math.top_k(A.score.stack(), k=1, sorted=True)
                    y_hat_score = y_hat_score[0]
                    y_hat_index = tf.gather_nd(A.indices.stack(), y_hat_score_index)
                    y_hat_prediction = tf.gather_nd(
                        math_util.pad_prediction_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
                        y_hat_score_index
                    )
                    y_hat_states = tf.gather_nd(A.states.stack(), y_hat_score_index)

                    # remove y_hat from A
                    remain_indices = tf.range(0, tf.shape(A.score.stack())[0], dtype=tf.int32)
                    remain_indices = tf.gather_nd(remain_indices, tf.where(tf.not_equal(remain_indices, y_hat_score_index[0])))
                    remain_indices = tf.expand_dims(remain_indices, axis=-1)
                    A = BeamHypothesis(
                        score=A.score.unstack(tf.gather_nd(A.score.stack(), remain_indices)),
                        indices=A.indices.unstack(tf.gather_nd(A.indices.stack(), remain_indices)),
                        prediction=A.prediction.unstack(
                            tf.gather_nd(
                                math_util.pad_prediction_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
                                remain_indices
                            )
                        ),
                        states=A.states.unstack(tf.gather_nd(A.states.stack(), remain_indices)),
                    )
                    A_i = tf.cond(tf.equal(A_i, 0), true_fn=lambda: A_i, false_fn=lambda: A_i - 1)

                    ytu, new_states = self.decoder_inference(encoded=encoded_t, predicted=y_hat_index, states=y_hat_states)

                    def predict_condition(pred, A, A_i, B): return tf.less(pred, self.text_featurizer.num_classes)

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
                            scatter_index = math_util.count_non_blank(y_hat_prediction, blank=self.text_featurizer.blank)
                            updated_prediction = tf.tensor_scatter_nd_update(
                                y_hat_prediction,
                                indices=tf.reshape(scatter_index, [1, 1]),
                                updates=tf.expand_dims(pred, axis=-1)
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
                                A_i + 1
                            )

                        b_score, b_indices, b_prediction, b_states, \
                            a_score, a_indices, a_prediction, a_states, A_i = tf.cond(
                                tf.equal(pred, self.text_featurizer.blank), true_fn=true_fn, false_fn=false_fn)

                        B = BeamHypothesis(score=b_score, indices=b_indices, prediction=b_prediction, states=b_states)
                        A = BeamHypothesis(score=a_score, indices=a_indices, prediction=a_prediction, states=a_states)

                        return pred + 1, A, A_i, B

                    _, A, A_i, B = tf.while_loop(
                        predict_condition, predict_body,
                        loop_vars=[0, A, A_i, B],
                        parallel_iterations=parallel_iterations, swap_memory=swap_memory
                    )

                    return beam + 1, beam_width, A, A_i, B

                _, _, A, A_i, B = tf.while_loop(
                    beam_condition, beam_body,
                    loop_vars=[0, beam_width, A, A_i, B],
                    parallel_iterations=parallel_iterations, swap_memory=swap_memory
                )

                return time + 1, total, B

            _, _, B = tf.while_loop(
                condition, body,
                loop_vars=[0, total, B],
                parallel_iterations=parallel_iterations, swap_memory=swap_memory
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

    def make_tflite_function(self, timestamp: bool = False):
        tflite_func = self.recognize_tflite_with_timestamp if timestamp else self.recognize_tflite
        return tf.function(
            tflite_func,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32)
            ]
        )
