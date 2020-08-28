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

import collections
import tensorflow as tf

from . import Model
from ..utils.utils import shape_list, get_shape_invariants, get_float_spec
from ..featurizers.speech_featurizers import TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from .layers.embedding import Embedding

Hypothesis = collections.namedtuple(
    "Hypothesis",
    ("score", "prediction", "states")
)

BeamHypothesis = collections.namedtuple(
    "BeamHypothesis",
    ("score", "prediction", "states", "lm_states")
)


class TransducerPrediction(tf.keras.layers.Layer):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = Embedding(vocabulary_size, embed_dim,
                               regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        self.lstms = []
        # lstms units must equal (for using beam search)
        for i in range(num_lstms):
            lstm = tf.keras.layers.LSTM(
                units=lstm_units, return_sequences=True,
                name=f"{name}_lstm_{i}", return_state=True,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            self.lstms.append(lstm)

    def get_initial_state(self):
        memory_states = []
        for i in range(len(self.lstms)):
            memory_states.append(
                self.lstms[i].get_initial_state(
                    tf.zeros([1, 1, 1], dtype=tf.float32)
                )
            )
        return memory_states

    def call(self, inputs, training=False):
        # inputs has shape [B, U]
        # use tf.gather_nd instead of tf.gather for tflite conversion
        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        for lstm in self.lstms:
            outputs, _, _ = lstm(outputs, training=training)
        return outputs

    def inference(self, inputs, states):
        # inputs has shape [B, U]
        outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        n_memory_states = []
        for i, lstm in enumerate(self.lstms):
            outputs = lstm(outputs, training=False, initial_state=states[i])
            new_memory_states = outputs[1:]
            outputs = outputs[0]
            n_memory_states.append(new_memory_states)
        # return shapes [B, T, P], [([B, P], [B, P]), ...]
        return outputs, n_memory_states

    def get_config(self):
        conf = super(TransducerPrediction, self).get_config()
        conf.update(self.embed.get_config())
        conf.update(self.do.get_config())
        for lstm in self.lstms:
            conf.update(lstm.get_config())
        return conf


class TransducerJoint(tf.keras.layers.Layer):
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

    def call(self, inputs, training=False):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        enc_out = tf.expand_dims(enc_out, axis=2)
        pred_out = tf.expand_dims(pred_out, axis=1)
        enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, 1, E] => [B, T, 1, V]
        pred_out = self.ffn_pred(pred_out, training=training)  # [B, 1, U, P] => [B, 1, U, V]
        outputs = tf.nn.tanh(enc_out + pred_out)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(TransducerJoint, self).get_config()
        conf.update(self.ffn_enc.get_config())
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
                 num_lstms: int = 1,
                 lstm_units: int = 320,
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
            num_lstms=num_lstms,
            lstm_units=lstm_units,
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
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        self([inputs, pred], training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def call(self, inputs, training=False):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            predicted: predicted sequence of character ids, in shape [B, U]
            training: python boolean
            **kwargs: sth else

        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, predicted = inputs
        enc = self.encoder(features, training=training)
        pred = self.predict_net(predicted, training=training)
        outputs = self.joint_net([enc, pred], training=training)
        return outputs

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

    @tf.function
    def recognize(self, features: tf.Tensor) -> tf.Tensor:
        b_i = tf.constant(0, dtype=tf.int32)

        B = shape_list(features)[0]

        decoded = tf.constant([], dtype=tf.string)

        def _cond(b_i, B, features, decoded): return tf.less(b_i, B)

        def _body(b_i, B, features, decoded):
            yseq = self.perform_greedy(features[b_i])
            yseq = self.text_featurizer.iextract(tf.expand_dims(yseq.prediction, axis=0))
            decoded = tf.concat([decoded, yseq], axis=0)
            return b_i + 1, B, features, decoded

        _, _, _, decoded = tf.while_loop(
            _cond,
            _body,
            loop_vars=(b_i, B, features, decoded),
            swap_memory=True,
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([]),
                get_shape_invariants(features),
                tf.TensorShape([None])
            )
        )

        return decoded

    def recognize_tflite(self,
                         signal: tf.Tensor,
                         prediction: Hypothesis) -> tf.Tensor:
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            prediction
        """
        features = self.speech_featurizer.tf_extract(signal)
        hypothesis = self.perform_greedy(features, prediction)
        transcript = self.text_featurizer.index2upoints(hypothesis.prediction)
        return (
            transcript,
            (
                hypothesis.score,
                tf.expand_dims(hypothesis.prediction[-1], axis=0),
                hypothesis.states
            )
        )

    def perform_greedy(self,
                       features: tf.Tensor,
                       prediction: tuple = None) -> tf.Tensor:
        with tf.name_scope("perform_greedy"):
            new_hyps = Hypothesis(
                tf.constant(0.0, dtype=tf.float32),
                tf.constant([self.text_featurizer.blank], dtype=tf.int32),
                self.predict_net.get_initial_state()
            )

            if prediction is not None: new_hyps = Hypothesis(*prediction)

            enc = self.encoder(tf.expand_dims(features, axis=0), training=False)  # [1, T, E]
            enc = tf.squeeze(enc, axis=0)  # [T, E]

            T = tf.cast(shape_list(enc)[0], dtype=tf.int32)

            i = tf.constant(0, dtype=tf.int32)

            def _cond(enc, i, new_hyps, T): return tf.less(i, T)

            def _body(enc, i, new_hyps, T):
                hi = tf.reshape(enc[i], [1, 1, -1])  # [1, 1, E]
                y, new_states = self.predict_net.inference(
                    inputs=tf.reshape(new_hyps.prediction[-1], [1, 1]),  # [1, 1]
                    states=new_hyps.states
                )  # [1, 1, P], [1, P], [1, P]
                # [1, 1, E] + [1, 1, P] => [1, 1, 1, V]
                ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))
                ytu = tf.squeeze(ytu, axis=None)  # [1, 1, 1, V] => [V]
                n_predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                def return_no_blank():
                    return Hypothesis(
                        new_hyps.score + ytu[n_predict],
                        tf.concat([new_hyps.prediction, [n_predict]], axis=0),
                        new_states
                    )

                hyps = tf.cond(
                    n_predict != self.text_featurizer.blank,
                    true_fn=return_no_blank,
                    false_fn=lambda: new_hyps
                )

                return enc, i + 1, hyps, T

            _, _, new_hyps, _ = tf.while_loop(
                _cond,
                _body,
                loop_vars=(enc, i, new_hyps, T),
                swap_memory=True,
                shape_invariants=(
                    get_shape_invariants(enc),
                    tf.TensorShape([]),
                    Hypothesis(
                        tf.TensorShape([]),
                        tf.TensorShape([None]),
                        tf.nest.map_structure(get_shape_invariants,
                                              self.predict_net.get_initial_state())
                    ),
                    tf.TensorShape([])
                )
            )

            return new_hyps

    @tf.function
    def recognize_beam(self,
                       features: tf.Tensor,
                       lm: bool = False) -> tf.Tensor:
        if lm: return self.recognize(features)

        b_i = tf.constant(0, dtype=tf.int32)

        B = shape_list(features)[0]

        decoded = tf.constant([], dtype=tf.string)

        def _cond(b_i, B, features, decoded, lm): return tf.less(b_i, B)

        def _body(b_i, B, features, decoded, lm):
            yseq = tf.py_function(self.perform_beam_search,
                                  inp=[tf.expand_dims(features[b_i], axis=0), lm],
                                  Tout=tf.int32)
            yseq = self.text_featurizer.iextract(yseq)
            decoded = tf.concat([decoded, yseq], axis=0)
            return b_i + 1, B, features, decoded, lm

        _, _, _, decoded, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(b_i, B, features, decoded, lm),
            swap_memory=True,
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([]),
                get_shape_invariants(features),
                tf.TensorShape([None]),
                tf.TensorShape([])
            )
        )

        return decoded

    def perform_beam_search(self,
                            features: tf.Tensor,
                            lm: bool = False) -> tf.Tensor:
        beam_width = self.text_featurizer.decoder_config["beam_width"]
        norm_score = self.text_featurizer.decoder_config["norm_score"]
        lm = lm.numpy()

        kept_hyps = [
            BeamHypothesis(
                score=0.0,
                prediction=[self.text_featurizer.blank],
                states=self.predict_net.get_initial_state(),
                lm_states=None
            )
        ]

        enc = tf.squeeze(self.encoder(features, training=False), axis=0)  # [T, E]

        B = kept_hyps

        for i in range(shape_list(enc)[0]):  # [E]
            hi = tf.reshape(enc[i], [1, 1, -1])  # [1, 1, E]
            A = B  # A = hyps
            B = []

            while True:
                y_hat = max(A, key=lambda x: x.score)
                A.remove(y_hat)

                y, new_states = self.predict_net.inference(
                    inputs=tf.reshape(y_hat.prediction[-1], [1, 1]),
                    states=y_hat.states
                )
                ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))  # [1, 1, 1, V]
                ytu = tf.squeeze(ytu)

                if lm and self.text_featurizer.scorer:
                    lm_state, lm_score = self.text_featurizer.scorer(y_hat)

                for k in range(self.text_featurizer.num_classes):
                    beam_hyp = BeamHypothesis(
                        score=(y_hat.score + float(ytu[k].numpy())),
                        prediction=y_hat.prediction,
                        states=y_hat.states,
                        lm_states=y_hat.lm_states
                    )

                    if k == self.text_featurizer.blank:
                        B.append(beam_hyp)
                    else:
                        beam_hyp = BeamHypothesis(
                            score=beam_hyp.score,
                            prediction=(beam_hyp.prediction + [int(k)]),
                            states=new_states,
                            lm_states=beam_hyp.lm_states
                        )

                        if lm and self.text_featurizer.scorer:
                            beam_hyp = BeamHypothesis(
                                score=(beam_hyp.score + lm_score),
                                prediction=beam_hyp.prediction,
                                states=new_states,
                                lm_states=lm_state
                            )

                        A.append(beam_hyp)

                if len(B) >= beam_width: break

        if norm_score:
            kept_hyps = sorted(B, key=lambda x: x.score / len(x.prediction),
                               reverse=True)[:beam_width]
        else:
            kept_hyps = sorted(B, key=lambda x: x.score, reverse=True)[:beam_width]

        return tf.convert_to_tensor(kept_hyps[0].prediction, dtype=tf.int32)[None, ...]

    def make_tflite_function(self, greedy: bool = True):
        return tf.function(
            self.recognize_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                (
                    tf.TensorSpec([], dtype=tf.float32),
                    tf.TensorSpec([1], dtype=tf.int32),
                    tf.nest.map_structure(get_float_spec, self.predict_net.get_initial_state())
                )
            ]
        )

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
