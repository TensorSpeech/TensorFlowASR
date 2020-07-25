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

import os
import collections
import tensorflow as tf

from ..utils.utils import shape_list, get_shape_invariants, merge_repeated
from ..featurizers.speech_featurizers import TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer

Hypotheses = collections.namedtuple(
    "Hypotheses",
    ("scores", "yseqs", "p_memory_states")
)


class TransducerPrediction(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 name="transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = tf.keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embed_dim, mask_zero=True)
        self.do = tf.keras.layers.Dropout(embed_dropout)
        self.lstms = []
        # lstms units must equal (for using beam search)
        for i in range(num_lstms):
            lstm = tf.keras.layers.LSTM(units=lstm_units,
                                        return_sequences=True, return_state=True)
            self.lstms.append(lstm)

    def get_initial_state(self, batch_size):
        memory_states = []
        for i in range(len(self.lstms)):
            memory_states.append(self.lstms[i].get_initial_state(tf.zeros([batch_size, 1, 1])))
        return memory_states

    @tf.function(experimental_relax_shapes=True)
    def call(self,
             inputs,
             training=False,
             p_memory_states=None,
             **kwargs):
        # inputs has shape [B, U]
        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        if p_memory_states is None:  # Zeros mean no initial_state
            p_memory_states = self.get_initial_state(shape_list(outputs)[0])
        n_memory_states = []
        for i, lstm in enumerate(self.lstms):
            outputs = lstm(outputs, training=training, initial_state=p_memory_states[i])
            new_memory_states = outputs[1:]
            outputs = outputs[0]
            n_memory_states.append(new_memory_states)

        # return shapes [B, T, P], ([num_lstms, B, P], [num_lstms, B, P]) if using lstm
        return outputs, n_memory_states

    def get_config(self):
        conf = super(TransducerPrediction, self).get_config()
        conf.update(self.embed.get_config())
        conf.update(self.do.get_config())
        for lstm in self.lstms:
            conf.update(lstm.get_config())
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 name="tranducer_joint",
                 **kwargs):
        super(TransducerJoint, self).__init__(name=name, **kwargs)
        self.ffn_enc = tf.keras.layers.Dense(joint_dim)
        self.ffn_pred = tf.keras.layers.Dense(joint_dim)
        self.ffn_out = tf.keras.layers.Dense(vocabulary_size)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc, pred = inputs
        enc_out = self.ffn_enc(enc, training=training)  # [B, T ,E] => [B, T, V]
        pred_out = self.ffn_pred(pred, training=training)  # [B, U, P] => [B, U, V]
        # => [B, T, U, V]
        outputs = tf.nn.tanh(tf.expand_dims(enc_out, axis=2) + tf.expand_dims(pred_out, axis=1))
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(TransducerJoint, self).get_config()
        conf.update(self.ffn_enc.get_config())
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        return conf


class Transducer(tf.keras.Model):
    """ Transducer Model Warper """

    def __init__(self,
                 encoder: tf.keras.Model,
                 vocabulary_size: int,
                 blank: int = 0,
                 embed_dim: int = 512,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 320,
                 joint_dim: int = 1024,
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
            name=f"{name}_prediction"
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            name=f"{name}_joint"
        )
        self.kept_hyps = None

    def _build(self, sample_shape):  # Call on real data for building model
        features = tf.random.normal(shape=sample_shape)
        predicted = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        self([features, predicted], training=True)

    def save_seperate(self, path_to_dir: str):
        self.encoder.save(os.path.join(path_to_dir, "encoder"))
        self.predict_net.save(os.path.join(path_to_dir, "prediction"))
        self.joint_net.save(os.path.join(path_to_dir, "joint"))

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    @tf.function(experimental_relax_shapes=True)
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
        pred, _ = self.predict_net(predicted, training=training)
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

    @tf.function(experimental_relax_shapes=True)
    def recognize(self, features: tf.Tensor) -> tf.Tensor:
        decoded = self.perform_greedy(features, streaming=False)
        return self.text_featurizer.iextract(decoded)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None], dtype=tf.float32),
        ]
    )
    def recognize_tflite(self, signal: tf.Tensor) -> tf.Tensor:
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
        """
        features = self.speech_featurizer.tf_extract(signal)
        features = tf.expand_dims(features, axis=0)
        indices = self.perform_greedy(features, streaming=True)
        transcript = self.text_featurizer.index2upoints(indices)
        return tf.squeeze(transcript, axis=0)

    @tf.function(experimental_relax_shapes=True)
    def perform_greedy(self,
                       features: tf.Tensor,
                       streaming: bool = False) -> tf.Tensor:
        B = shape_list(features)[0]

        if self.kept_hyps is None:
            new_hyps = Hypotheses(
                tf.zeros([B], dtype=tf.float32),
                self.text_featurizer.blank * tf.ones([B, 1], dtype=tf.int32),
                self.predict_net.get_initial_state(B)
            )
        else:
            new_hyps = self.kept_hyps

        enc = self.encoder(features, training=False)  # [B, T, E]

        T = tf.cast(shape_list(enc)[1], dtype=tf.int32)

        i = tf.constant(0, dtype=tf.int32)

        def _cond(enc, i, new_hyps, T, B): return tf.less(i, T)

        def _body(enc, i, new_hyps, T, B):
            hi = tf.reshape(enc[:, i], [B, 1, -1])  # [B, 1, E]
            y, n_memory_states = self.predict_net(
                inputs=tf.reshape(new_hyps[1][:, -1], [B, 1]),  # [B, 1]
                p_memory_states=new_hyps[2],
                training=False
            )  # [B, 1, P], [B, P], [B, P]
            # [B, 1, E] + [B, 1, P] => [B, 1, 1, V]
            ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))
            ytu = tf.squeeze(ytu, axis=1)  # [B, 1, 1, V] => [B, 1, V]
            n_predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax [B, 1]
            ytu = tf.squeeze(ytu, axis=1)  # [B, V]
            indices = tf.stack([tf.range(B, dtype=tf.int32),
                                tf.squeeze(n_predict, axis=-1)], axis=-1)  # [B, 2]

            yseq = tf.concat([new_hyps[1], n_predict], axis=-1)
            yseq = merge_repeated(yseq, self.text_featurizer.blank)

            hyps = Hypotheses(
                new_hyps[0] + tf.gather_nd(ytu, indices),
                yseq,
                n_memory_states,
            )

            return enc, i + 1, hyps, T, B

        _, _, new_hyps, _, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(enc, i, new_hyps, T, B),
            shape_invariants=(
                tf.TensorShape([None, None, None]),
                tf.TensorShape([]),
                Hypotheses(
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.nest.map_structure(get_shape_invariants, new_hyps[-1])
                ),
                tf.TensorShape([]),
                tf.TensorShape([])
            )
        )

        if streaming: self.kept_hyps = new_hyps

        return new_hyps[1]

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None, None, None], dtype=tf.float32),  # features
            tf.TensorSpec([], dtype=tf.bool),  # lm
            tf.TensorSpec([], dtype=tf.bool)  # streaming
        ]
    )
    def recognize_beam(self,
                       features: tf.Tensor,
                       lm: bool = False,
                       streaming: bool = False) -> tf.Tensor:
        # return self.recognize(features)
        def map_fn(elem):
            return tf.py_function(
                self.perform_beam_search,
                inp=[tf.expand_dims(elem, axis=0), lm, streaming],
                Tout=tf.int32
            )

        decoded = tf.map_fn(map_fn, features, dtype=tf.int32)
        return self.text_featurizer.iextract(decoded)

#    @tf.function(
#        experimental_relax_shapes=True,
#        input_signature=[
#            tf.TensorSpec([None], dtype=tf.float32),  # signal
#            tf.TensorSpec([], dtype=tf.bool),  # lm
#            tf.TensorSpec([], dtype=tf.bool)  # streaming
#        ]
#    )
#    def recognize_beam_tflite(self,
#                              signal,
#                              lm=False,
#                              streaming=False):
#        def func(signal, lm, streaming):
#            features = self.speech_featurizer.extract(signal)
#            transcript = self.perform_beam_search(
#                tf.expand_dims(features, axis=0), lm, streaming)
#            transcript = self.text_featurizer.iextract(tf.expand_dims(transcript, axis=0))
#            return tf.squeeze(transcript, axis=0)
#
#        return tf.py_function(func, inp=[signal, lm, streaming], Tout=tf.string)

    def perform_beam_search(self,
                            features: tf.Tensor,
                            lm: bool = False,
                            streaming: bool = False) -> tf.Tensor:
        beam_width = self.text_featurizer.decoder_config["beam_width"]
        norm_score = self.text_featurizer.decoder_config["norm_score"]
        lm = lm.numpy()
        streaming = streaming.numpy()
        if self.kept_hyps is None:
            self.kept_hyps = [
                {
                    "score": 0.0,
                    "yseq": [self.text_featurizer.blank],
                    "p_memory_states": None,
                    "lm_state": None
                }
            ]

        if not streaming:
            self.kept_hyps = [
                {
                    "score": 0.0,
                    "yseq": [self.text_featurizer.blank],
                    "p_memory_states": None,
                    "lm_state": None
                }
            ]

        enc = tf.squeeze(self.encoder(features, training=False), axis=0)  # [T, E]

        kept_hyps = self.kept_hyps

        for i in range(shape_list(enc)[0]):  # [E]
            hi = tf.reshape(enc[i], [1, 1, -1])  # [1, 1, E]
            hyps = kept_hyps  # A = hyps
            kept_hyps = []

            while True:
                new_hyps = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyps)

                y, n_memory_states = self.predict_net(
                    inputs=tf.reshape(new_hyps["yseq"][-1], [1, 1]),
                    p_memory_states=new_hyps["p_memory_states"],
                    training=False
                )
                ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))  # [1, 1, 1, V]

                if lm and self.text_featurizer.scorer:
                    lm_state, lm_score = self.text_featurizer.scorer(new_hyps)

                for k in range(self.text_featurizer.num_classes):
                    beam_hyp = new_hyps
                    beam_hyp["score"] += tf.squeeze(ytu)[k].numpy()

                    if k == self.text_featurizer.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["yseq"] += [int(k)]
                        beam_hyp["p_memory_states"] = n_memory_states

                        if lm and self.text_featurizer.scorer:
                            beam_hyp["lm_state"] = lm_state
                            beam_hyp["score"] += lm_score

                        hyps.append(beam_hyp)

                if len(kept_hyps) >= beam_width:
                    break

        # get nbest hyp using quick sort
        if norm_score:
            kept_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]),
                reverse=True)[:beam_width]
        else:
            kept_hyps = sorted(
                kept_hyps, key=lambda x: x["score"],
                reverse=True)[:beam_width]

        if streaming: self.kept_hyps = kept_hyps

        return tf.convert_to_tensor(kept_hyps[0]["yseq"], dtype=tf.int32)

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
