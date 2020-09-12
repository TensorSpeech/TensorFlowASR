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

from .conformer import ConformerEncoder
from .transducer import Transducer
from ..featurizers.speech_featurizers import SpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils.utils import get_shape_invariants

L2 = tf.keras.regularizers.l2(1e-6)


class MultiConformers(Transducer):
    def __init__(self,
                 subsampling: dict,
                 dmodel: int = 144,
                 vocabulary_size: int = 29,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 mha_type: str = "relmha",
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
        super(MultiConformers, self).__init__(
            encoder=None,
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
        )
        self.encoder_lms = ConformerEncoder(
            subsampling=subsampling,
            dmodel=dmodel,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            mha_type=mha_type,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder_lms"
        )
        self.encoder_lgs = ConformerEncoder(
            subsampling=subsampling,
            dmodel=dmodel,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            mha_type=mha_type,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder_lgs"
        )
        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")
        self.encoder_joint = tf.keras.layers.Dense(dmodel, name=f"{name}_enc_joint")
        self.time_reduction_factor = self.encoder_lms.conv_subsampling.time_reduction_factor

    def _build(self, lms_shape, lgs_shape):
        lms = tf.keras.Input(shape=lms_shape, dtype=tf.float32)
        lgs = tf.keras.Input(shape=lgs_shape, dtype=tf.float32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        self([lms, lgs, pred], training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder_lms.summary(line_length=line_length, **kwargs)
        super(MultiConformers, self).summary(line_length=line_length, **kwargs)

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

    def encoder_inference(self, features):
        """Infer function for encoder (or encoders)

        Args:
            features (list or tuple): 2 features, each has shape [T, F, C]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            lms, lgs = features
            lms = tf.expand_dims(lms, axis=0)
            lgs = tf.expand_dims(lgs, axis=0)

            enc_lms_out = self.encoder_lms(lms, training=False)
            enc_lgs_out = self.encoder_lgs(lgs, training=False)

            outputs = self.concat([enc_lms_out, enc_lgs_out], training=False)
            outputs = self.encoder_joint(outputs, training=False)

            return tf.squeeze(outputs, axis=0)

    def get_config(self):
        conf = self.encoder_lms.get_config()
        conf.update(self.encoder_lgs.get_config())
        conf.update(self.concat.get_config())
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self, features):
        lms, lgs = features

        total = tf.shape(lms)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, lms, lgs, decoded): return tf.less(batch, total)

        def body(batch, total, lms, lgs, decoded):
            yseq = self.perform_greedy(
                [lms[batch], lgs[batch]],
                predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                states=self.predict_net.get_initial_state(),
                swap_memory=True
            )
            yseq = self.text_featurizer.iextract(tf.expand_dims(yseq.prediction, axis=0))
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, lms, lgs, decoded

        batch, total, lms, lgs, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, lms, lgs, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(lms),
                get_shape_invariants(lgs),
                tf.TensorShape([None])
            )
        )

        return decoded

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self, features, lm=False):
        lms, lgs = features

        total = tf.shape(lms)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, lms, lgs, decoded): return tf.less(batch, total)

        def body(batch, total, lms, lgs, decoded):
            yseq = tf.py_function(self.perform_beam_search,
                                  inp=[[lms[batch], lgs[batch]], lm],
                                  Tout=tf.int32)
            yseq = self.text_featurizer.iextract(yseq)
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, lms, lgs, decoded

        batch, total, lms, lgs, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, lms, lgs, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(lms),
                get_shape_invariants(lgs),
                tf.TensorShape([None]),
            )
        )

        return decoded
