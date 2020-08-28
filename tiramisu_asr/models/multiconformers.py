# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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

import tensorflow as tf

from . import Model
from .conformer import ConformerEncoder
from .transducer import TransducerJoint, TransducerPrediction, Hypothesis, BeamHypothesis
from ..featurizers.speech_featurizers import SpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils.utils import get_shape_invariants, shape_list

L2 = tf.keras.regularizers.L2(1e-6)


class MultiConformers(Model):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 320,
                 joint_dim: int = 1024,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="multi-conformers",
                 **kwargs):
        super(MultiConformers, self).__init__(name=name, **kwargs)
        self.time_reduction_factor = reduction_factor
        self.encoder_lms = ConformerEncoder(
            dmodel=dmodel,
            reduction_factor=reduction_factor,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder_lms"
        )
        self.encoder_lgs = ConformerEncoder(
            dmodel=dmodel,
            reduction_factor=reduction_factor,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder_lgs"
        )
        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")
        self.encoder_joint = tf.keras.layers.Dense(dmodel, name=f"{name}_enc_joint")
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

    def _build(self, lms_shape, lgs_shape):
        lms = tf.keras.Input(shape=lms_shape, dtype=tf.float32)
        lgs = tf.keras.Input(shape=lgs_shape, dtype=tf.float32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        self([lms, lgs, pred], training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder_lms.summary(line_length=line_length, **kwargs)
        super(MultiConformers, self).summary(line_length=line_length, **kwargs)

    def encoder(self, inputs, training=False):
        lms, lgs = inputs

        enc_lms_out = self.encoder_lms(lms, training=training)
        enc_lgs_out = self.encoder_lgs(lgs, training=training)

        outputs = self.concat([enc_lms_out, enc_lgs_out], training=training)
        return self.encoder_joint(outputs, training=training)

    def call(self, inputs, training=False):
        lms, lgs, predicted = inputs

        pred = self.predict_net(predicted, training=training)

        enc_lms_out = self.encoder_lms(lms, training=training)
        outputs_lms = self.joint_net([enc_lms_out, pred], training=training)

        enc_lgs_out = self.encoder_lgs(lgs, training=training)
        outputs_lgs = self.joint_net([enc_lgs_out, pred], training=training)

        enc = self.concat([enc_lms_out, enc_lgs_out], training=training)
        enc = self.encoder_joint(enc, training=training)
        outputs = self.joint_net([enc, pred], training=training)

        return outputs_lms, outputs, outputs_lgs

    def add_featurizers(self,
                        speech_featurizer_lms: SpeechFeaturizer,
                        speech_featurizer_lgs: SpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer_lms = speech_featurizer_lms
        self.speech_featurizer_lgs = speech_featurizer_lgs
        self.text_featurizer = text_featurizer

    @tf.function
    def recognize(self, lms: tf.Tensor, lgs: tf.Tensor) -> tf.Tensor:
        features = tf.stack([lms, lgs], axis=1)

        def _body(record):
            lms, lgs = tf.unstack(record, axis=0)
            decoded = self.perform_greedy(lms[None, ...], lgs[None, ...])
            decoded = self.text_featurizer.iextract(decoded)
            return tf.squeeze(decoded)

        return tf.map_fn(_body, features,
                         fn_output_signature=tf.TensorSpec([], dtype=tf.string))

    def perform_greedy(self,
                       lms: tf.Tensor,
                       lgs: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("perform_greedy"):
            new_hyps = Hypothesis(
                tf.constant(0.0, dtype=tf.float32),
                tf.constant([self.text_featurizer.blank], dtype=tf.int32),
                self.predict_net.get_initial_state()
            )

            enc = self.encoder([lms, lgs], training=False)  # [1, T, E]
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
                        tf.nest.map_structure(get_shape_invariants, new_hyps.states)
                    ),
                    tf.TensorShape([])
                )
            )

            return new_hyps.prediction[None, ...]

    @tf.function
    def recognize_beam(self,
                       lms: tf.Tensor,
                       lgs: tf.Tensor,
                       lm: bool = False) -> tf.Tensor:
        if lm: return self.recognize(lms, lgs)

        features = tf.stack([lms, lgs], axis=1)

        def _body(record):
            lms, lgs = tf.unstack(record, axis=0)
            decoded = tf.py_function(self.perform_beam_search,
                                     inp=[lms[None, ...], lgs[None, ...], lm, False],
                                     Tout=tf.int32)
            decoded = self.text_featurizer.iextract(decoded)
            return tf.squeeze(decoded)

        return tf.map_fn(_body, features,
                         fn_output_signature=tf.TensorSpec([], dtype=tf.string))

    def perform_beam_search(self,
                            lms: tf.Tensor,
                            lgs: tf.Tensor,
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

        enc = tf.squeeze(self.encoder([lms, lgs], training=False), axis=0)  # [T, E]

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

    def get_config(self):
        conf = self.encoder_lms.get_config()
        conf.update(self.encoder_lgs.get_config())
        conf.update(self.concat.get_config())
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
