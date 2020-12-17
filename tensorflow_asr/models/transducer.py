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
import tensorflow as tf

from . import Model
from ..utils.utils import get_rnn, shape_list
from ..featurizers.speech_featurizers import TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from .layers.embedding import Embedding

Hypothesis = collections.namedtuple(
    "Hypothesis",
    ("index", "prediction", "states")
)

BeamHypothesis = collections.namedtuple(
    "BeamHypothesis",
    ("score", "indices", "prediction", "states")
)


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
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = Embedding(vocabulary_size, embed_dim,
                               regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        # Initialize rnn layers
        RNN = get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units=rnn_units, return_sequences=True,
                name=f"{name}_lstm_{i}", return_state=True,
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
            mask = tf.sequence_mask(prediction_length)
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
            outputs = rnn["rnn"](outputs, training=False,
                                 initial_state=tf.unstack(states[i], axis=0))
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


class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="tranducer_joint",
                 **kwargs):
        super(TransducerJoint, self).__init__(name=name, **kwargs)
        self.ffn_enc = tf.keras.layers.Dense(
            joint_dim, name=f"{name}_enc",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ffn_pred = tf.keras.layers.Dense(
            joint_dim, use_bias=False, name=f"{name}_pred",
            kernel_regularizer=kernel_regularizer
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
        enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
        pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        enc_out = tf.expand_dims(enc_out, axis=2)
        pred_out = tf.expand_dims(pred_out, axis=1)
        outputs = tf.nn.tanh(enc_out + pred_out)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        return conf


class Transducer(Model):
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
                 joint_dim: int = 1024,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="transducer",
                 **kwargs):
        super(Transducer, self).__init__(name=name, **kwargs)
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
            name=f"{name}_prediction"
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_joint"
        )

    def _build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        input_length = tf.keras.Input(shape=[], dtype=tf.int32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        pred_length = tf.keras.Input(shape=[], dtype=tf.int32)
        self([inputs, input_length, pred, pred_length], training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: TFSpeechFeaturizer,
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

    def call(self, inputs, training=False, **kwargs):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            input_length: features time length in shape [B]
            prediction: predicted sequence of ids, in shape [B, U]
            prediction_length: predicted sequence of ids length in shape [B]
            training: python boolean
            **kwargs: sth else

        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, _, prediction, prediction_length = inputs
        enc = self.encoder(features, training=training, **kwargs)
        pred = self.predict_net([prediction, prediction_length], training=training, **kwargs)
        outputs = self.joint_net([enc, pred], training=training, **kwargs)
        return outputs

    def encoder_inference(self, features):
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

    def decoder_inference(self, encoded, predicted, states):
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
            ytu = tf.squeeze(ytu, axis=None)  # [1, 1, V] => [V]
            return ytu, new_states

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self, signals):
        """
        RNN Transducer Greedy decoding
        Args:
            signals (tf.Tensor): a batch of padded signals

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        def execute(signal: tf.Tensor):
            features = self.speech_featurizer.tf_extract(signal)
            encoded = self.encoder_inference(features)
            hypothesis = self.perform_greedy(
                encoded,
                predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                states=self.predict_net.get_initial_state(),
                swap_memory=True
            )
            transcripts = self.text_featurizer.iextract(tf.expand_dims(hypothesis.prediction, axis=0))
            return tf.squeeze(transcripts)  # reshape from [1] to []

        return tf.map_fn(execute, signals, fn_output_signature=tf.TensorSpec([], dtype=tf.string))

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
        hypothesis = self.perform_greedy(encoded, predicted, states, swap_memory=False)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return (
            transcript,
            hypothesis.prediction[-1],
            hypothesis.states
        )

    def perform_greedy(self, encoded, predicted, states, swap_memory=False):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = tf.shape(encoded)[0]
            # Initialize prediction with a blank
            # Prediction can not be longer than the encoded of audio plus blank
            prediction = tf.TensorArray(
                dtype=tf.int32,
                size=(total + 1),
                dynamic_size=False,
                element_shape=tf.TensorShape([]),
                clear_after_read=False
            )

            hypothesis = Hypothesis(
                index=tf.constant(0, dtype=tf.int32),
                prediction=prediction.write(0, predicted),
                states=states
            )

            def condition(time, total, encoded, hypothesis): return tf.less(time, total)

            def body(time, total, encoded, hypothesis):
                ytu, new_states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.expand_dims(time, axis=-1)),
                    predicted=hypothesis.prediction.read(hypothesis.index),
                    states=hypothesis.states
                )
                char = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                index, char, new_states = tf.cond(
                    tf.equal(char, self.text_featurizer.blank),
                    true_fn=lambda: (
                        hypothesis.index,
                        hypothesis.prediction.read(hypothesis.index),
                        hypothesis.states
                    ),
                    false_fn=lambda: (
                        hypothesis.index + 1,
                        char,
                        new_states
                    )
                )

                hypothesis = Hypothesis(
                    index=index,
                    prediction=hypothesis.prediction.write(index, char),
                    states=new_states
                )

                return time + 1, total, encoded, hypothesis

            time, total, encoded, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=(time, total, encoded, hypothesis),
                swap_memory=swap_memory
            )

            # Gather predicted sequence
            hypothesis = Hypothesis(
                index=hypothesis.index,
                prediction=tf.gather_nd(
                    params=hypothesis.prediction.stack(),
                    indices=tf.expand_dims(tf.range(hypothesis.index + 1), axis=-1)
                ),
                states=hypothesis.states
            )

            return hypothesis

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self, signals, lm=False):
        """
        RNN Transducer Beam Search
        Args:
            signals (tf.Tensor): a batch of padded signals
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        def execute(signal: tf.Tensor):
            features = self.speech_featurizer.tf_extract(signal)
            encoded = self.encoder_inference(features)
            hypothesis = self.perform_beam_search(encoded, lm)
            prediction = tf.map_fn(
                lambda x: tf.strings.to_number(x, tf.int32),
                tf.strings.split(hypothesis.prediction),
                fn_output_signature=tf.TensorSpec([], dtype=tf.int32)
            )
            transcripts = self.text_featurizer.iextract(tf.expand_dims(prediction, axis=0))
            return tf.squeeze(transcripts)  # reshape from [1] to []

        return tf.map_fn(execute, signals, fn_output_signature=tf.TensorSpec([], dtype=tf.string))

    def perform_beam_search(self, encoded, lm=False):
        with tf.device("/CPU:0"), tf.name_scope(f"{self.name}_beam_search"):
            beam_width = tf.cond(
                tf.less(self.text_featurizer.decoder_config.beam_width, self.text_featurizer.num_classes),
                true_fn=lambda: self.text_featurizer.decoder_config.beam_width,
                false_fn=lambda: self.text_featurizer.num_classes - 1
            )
            total = tf.shape(encoded)[0]

            def initialize_beam(dynamic=False):
                return BeamHypothesis(
                    score=tf.TensorArray(
                        dtype=tf.float32, size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic, element_shape=tf.TensorShape([]), clear_after_read=False
                    ),
                    indices=tf.TensorArray(
                        dtype=tf.int32, size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic, element_shape=tf.TensorShape([]), clear_after_read=False
                    ),
                    prediction=tf.TensorArray(
                        dtype=tf.string, size=beam_width if not dynamic else 0, dynamic_size=dynamic,
                        element_shape=tf.TensorShape([]), clear_after_read=False
                    ),
                    states=tf.TensorArray(
                        dtype=tf.float32, size=beam_width if not dynamic else 0,
                        dynamic_size=dynamic,
                        element_shape=tf.TensorShape(shape_list(self.predict_net.get_initial_state())),
                        clear_after_read=False
                    ),
                )

            B = initialize_beam()
            B = BeamHypothesis(
                score=B.score.write(0, 0.0),
                indices=B.indices.write(0, self.text_featurizer.blank),
                prediction=B.prediction.write(0, ''),
                states=B.states.write(0, self.predict_net.get_initial_state())
            )

            def condition(time, total, B): return tf.less(time, total)

            def body(time, total, B):
                A = initialize_beam(dynamic=True)
                A = BeamHypothesis(
                    score=A.score.unstack(B.score.stack()),
                    indices=A.indices.unstack(B.indices.stack()),
                    prediction=A.prediction.unstack(B.prediction.stack()),
                    states=A.states.unstack(B.states.stack()),
                )
                A_i = tf.constant(0, tf.int32)
                B = initialize_beam()

                encoded_t = tf.gather_nd(encoded, tf.expand_dims(time, axis=-1))

                def beam_condition(beam, beam_width, A, A_i, B): return tf.less(beam, beam_width)

                def beam_body(beam, beam_width, A, A_i, B):
                    y_hat_score, y_hat_score_index = tf.math.top_k(A.score.stack(), k=1)
                    y_hat_score = y_hat_score[0]
                    y_hat_index = tf.gather_nd(A.indices.stack(), y_hat_score_index)
                    y_hat_prediction = tf.gather_nd(A.prediction.stack(), y_hat_score_index)
                    y_hat_states = tf.gather_nd(A.states.stack(), y_hat_score_index)

                    ytu, new_states = self.decoder_inference(encoded=encoded_t, predicted=y_hat_index, states=y_hat_states)

                    def predict_condition(pred, A, A_i, B): return tf.less(pred, self.text_featurizer.num_classes)

                    def predict_body(pred, A, A_i, B):
                        new_score = y_hat_score + tf.gather_nd(ytu, tf.expand_dims(pred, axis=-1))
                        b_score, b_indices, b_prediction, b_states, \
                            a_score, a_indices, a_prediction, a_states, A_i = tf.cond(
                                tf.equal(pred, self.text_featurizer.blank),
                                true_fn=lambda: (
                                    B.score.write(beam, new_score),
                                    B.indices.write(beam, y_hat_index),
                                    B.prediction.write(beam, y_hat_prediction),
                                    B.states.write(beam, y_hat_states),
                                    A.score,
                                    A.indices,
                                    A.prediction,
                                    A.states,
                                    A_i,
                                ),
                                false_fn=lambda: (
                                    B.score,
                                    B.indices,
                                    B.prediction,
                                    B.states,
                                    A.score.write(A_i, new_score),
                                    A.indices.write(A_i, pred),
                                    A.prediction.write(A_i, tf.strings.reduce_join(
                                        [y_hat_prediction, tf.strings.format("{}", pred)], separator=" ")),
                                    A.states.write(A_i, new_states),
                                    A_i + 1
                                )
                            )
                        B = BeamHypothesis(score=b_score, indices=b_indices, prediction=b_prediction, states=b_states)
                        A = BeamHypothesis(score=a_score, indices=a_indices, prediction=a_prediction, states=a_states)
                        return pred + 1, A, A_i, B

                    _, A, A_i, B = tf.while_loop(predict_condition, predict_body, loop_vars=(0, A, A_i, B))

                    return beam + 1, beam_width, A, A_i, B

                _, _, A, A_i, B = tf.while_loop(beam_condition, beam_body, loop_vars=(0, beam_width, A, A_i, B))

                return time + 1, total, B

            _, _, B = tf.while_loop(condition, body, loop_vars=(0, total, B))

            scores = B.score.stack()
            if self.text_featurizer.decoder_config.norm_score:
                prediction_lengths = tf.strings.length(B.prediction.stack(), unit="UTF8_CHAR")
                scores /= tf.cast(prediction_lengths, dtype=scores.dtype)

            y_hat_score, y_hat_score_index = tf.math.top_k(scores, k=1)
            y_hat_score = y_hat_score[0]
            y_hat_index = tf.gather_nd(B.indices.stack(), y_hat_score_index)
            y_hat_prediction = tf.gather_nd(B.prediction.stack(), y_hat_score_index)
            y_hat_states = tf.gather_nd(B.states.stack(), y_hat_score_index)

            return Hypothesis(
                index=y_hat_index,
                prediction=y_hat_prediction,
                states=y_hat_states
            )

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(self, greedy: bool = True):
        return tf.function(
            self.recognize_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(),
                              dtype=tf.float32)
            ]
        )
