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

import collections
import tensorflow as tf

from .layers.subsampling import TimeReduction
from .transducer import Transducer, BeamHypothesis
from ..utils.utils import get_rnn, get_shape_invariants, merge_two_last_dims

Hypothesis = collections.namedtuple(
    "Hypothesis",
    ("index", "prediction", "encoder_states", "prediction_states")
)


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs): return merge_two_last_dims(inputs)


class StreamingTransducerBlock(tf.keras.Model):
    def __init__(self,
                 reduction_factor: int = 0,
                 dmodel: int = 640,
                 rnn_type: str = "lstm",
                 rnn_units: int = 2048,
                 layer_norm: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(StreamingTransducerBlock, self).__init__(**kwargs)

        if reduction_factor > 0:
            self.reduction = TimeReduction(reduction_factor, name=f"{self.name}_reduction")
        else:
            self.reduction = None

        RNN = get_rnn(rnn_type)
        self.rnn = RNN(
            units=rnn_units, return_sequences=True,
            name=f"{self.name}_rnn", return_state=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

        if layer_norm:
            self.ln = tf.keras.layers.LayerNormalization(name=f"{self.name}_ln")
        else:
            self.ln = None

        self.projection = tf.keras.layers.Dense(
            dmodel, name=f"{self.name}_projection",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    def call(self, inputs, training=False):
        outputs = inputs
        if self.reduction is not None:
            outputs = self.reduction(outputs)
        outputs = self.rnn(outputs, training=training)
        outputs = outputs[0]
        if self.ln is not None:
            outputs = self.ln(outputs, training=training)
        outputs = self.projection(outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        outputs = inputs
        if self.reduction is not None:
            outputs = self.reduction(outputs)
        outputs = self.rnn(outputs, training=False, initial_state=states)
        new_states = tf.stack(outputs[1:], axis=0)
        outputs = outputs[0]
        if self.ln is not None:
            outputs = self.ln(outputs, training=False)
        outputs = self.projection(outputs, training=False)
        return outputs, new_states

    def get_config(self):
        conf = {}
        if self.reduction is not None:
            conf.update(self.reduction.get_config())
        conf.update(self.rnn.get_config())
        if self.ln is not None:
            conf.update(self.ln.get_config())
        conf.update(self.projection.get_config())
        return conf


class StreamingTransducerEncoder(tf.keras.Model):
    def __init__(self,
                 reductions: dict = {0: 3, 1: 2},
                 dmodel: int = 640,
                 nlayers: int = 8,
                 rnn_type: str = "lstm",
                 rnn_units: int = 2048,
                 layer_norm: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(StreamingTransducerEncoder, self).__init__(**kwargs)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.blocks = [
            StreamingTransducerBlock(
                reduction_factor=reductions.get(i, 0),  # key is index, value is the factor
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}"
            ) for i in range(nlayers)
        ]

        self.time_reduction_factor = 1
        for i in range(nlayers):
            reduction_factor = reductions.get(i, 0)
            if reduction_factor > 0: self.time_reduction_factor *= reduction_factor

    def get_initial_state(self):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, P]
        """
        states = []
        for block in self.blocks:
            states.append(
                tf.stack(
                    block.rnn.get_initial_state(
                        tf.zeros([1, 1, 1], dtype=tf.float32)
                    ), axis=0
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False):
        outputs = self.reshape(inputs)
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for encoder network

        Args:
            inputs (tf.Tensor): shape [1, T, F, C]
            states (tf.Tensor): shape [num_lstms, 1 or 2, 1, P]

        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [num_lstms, 1 or 2, 1, P]
        """
        outputs = self.reshape(inputs)
        new_states = []
        for i, block in enumerate(self.blocks):
            outputs, block_states = block.recognize(outputs, states=tf.unstack(states[i], axis=0))
            new_states.append(block_states)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.reshape.get_config()
        for block in self.blocks: conf.update(block.get_config())
        return conf


class StreamingTransducer(Transducer):
    def __init__(self,
                 vocabulary_size: int,
                 encoder_reductions: dict = {0: 3, 1: 2},
                 encoder_dmodel: int = 640,
                 encoder_nlayers: int = 8,
                 encoder_rnn_type: str = "lstm",
                 encoder_rnn_units: int = 2048,
                 encoder_layer_norm: bool = True,
                 prediction_embed_dim: int = 320,
                 prediction_embed_dropout: float = 0,
                 prediction_num_rnns: int = 2,
                 prediction_rnn_units: int = 2048,
                 prediction_rnn_type: str = "lstm",
                 prediction_layer_norm: bool = True,
                 prediction_projection_units: int = 640,
                 joint_dim: int = 640,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 name = "StreamingTransducer",
                 **kwargs):
        super(StreamingTransducer, self).__init__(
            encoder=StreamingTransducerEncoder(
                reductions=encoder_reductions,
                dmodel=encoder_dmodel,
                nlayers=encoder_nlayers,
                rnn_type=encoder_rnn_type,
                rnn_units=encoder_rnn_units,
                layer_norm=encoder_layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_encoder"
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=prediction_embed_dim,
            embed_dropout=prediction_embed_dropout,
            num_rnns=prediction_num_rnns,
            rnn_units=prediction_rnn_units,
            rnn_type=prediction_rnn_type,
            layer_norm=prediction_layer_norm,
            projection_units=prediction_projection_units,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name, **kwargs
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor

    def summary(self, line_length=None, **kwargs):
        for block in self.encoder.blocks:
            block.summary(line_length=line_length, **kwargs)
        super(StreamingTransducer, self).summary(line_length=line_length, **kwargs)

    def encoder_inference(self, features, states):
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
    def recognize(self, features):
        total = tf.shape(features)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, features, decoded): return tf.less(batch, total)

        def body(batch, total, features, decoded):
            yseq = self.perform_greedy(
                features[batch],
                predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                encoder_states=self.encoder.get_initial_state(),
                prediction_states=self.predict_net.get_initial_state(),
                swap_memory=True
            )
            yseq = self.text_featurizer.iextract(tf.expand_dims(yseq.prediction, axis=0))
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, features, decoded

        batch, total, features, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, features, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(features),
                tf.TensorShape([None])
            )
        )

        return decoded

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
        hypothesis = self.perform_greedy(features, predicted,
                                         encoder_states, prediction_states, swap_memory=False)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return (
            transcript,
            hypothesis.prediction[-1],
            hypothesis.encoder_states,
            hypothesis.prediction_states,
        )

    def perform_greedy(self, features, predicted,
                       encoder_states, prediction_states, swap_memory=False):
        with tf.name_scope(f"{self.name}_greedy"):
            encoded, new_encoder_states = self.encoder_inference(features, encoder_states)
            # Initialize prediction with a blank
            # Prediction can not be longer than the encoded of audio plus blank
            prediction = tf.TensorArray(
                dtype=tf.int32,
                size=(tf.shape(encoded)[0] + 1),
                dynamic_size=False,
                element_shape=tf.TensorShape([]),
                clear_after_read=False
            )
            time = tf.constant(0, dtype=tf.int32)
            total = tf.shape(encoded)[0]

            hypothesis = Hypothesis(
                index=tf.constant(0, dtype=tf.int32),
                prediction=prediction.write(0, predicted),
                encoder_states=new_encoder_states,
                prediction_states=prediction_states
            )

            def condition(time, total, encoded, hypothesis): return tf.less(time, total)

            def body(time, total, encoded, hypothesis):
                ytu, new_states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.expand_dims(time, axis=-1)),
                    predicted=hypothesis.prediction.read(hypothesis.index),
                    states=hypothesis.prediction_states
                )
                char = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                index, char, new_states = tf.cond(
                    tf.equal(char, self.text_featurizer.blank),
                    true_fn=lambda: (
                        hypothesis.index,
                        hypothesis.prediction.read(hypothesis.index),
                        hypothesis.prediction_states
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
                    encoder_states=new_encoder_states,
                    prediction_states=new_states
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
                encoder_states=hypothesis.encoder_states,
                prediction_states=hypothesis.prediction_states
            )

            return hypothesis

    # -------------------------------- BEAM SEARCH -------------------------------------

    def perform_beam_search(self, features, lm=False):
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

        enc, _ = self.encoder_inference(features, self.encoder.get_initial_state())
        total = tf.shape(enc)[0].numpy()

        B = kept_hyps

        for i in range(total):  # [E]
            A = B  # A = hyps
            B = []

            while True:
                y_hat = max(A, key=lambda x: x.score)
                A.remove(y_hat)

                ytu, new_states = self.decoder_inference(
                    encoded=tf.gather_nd(enc, tf.expand_dims(i, axis=-1)),
                    predicted=y_hat.prediction[-1],
                    states=y_hat.states
                )

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

                if len(B) > beam_width: break

        if norm_score:
            kept_hyps = sorted(B, key=lambda x: x.score / len(x.prediction),
                               reverse=True)[:beam_width]
        else:
            kept_hyps = sorted(B, key=lambda x: x.score, reverse=True)[:beam_width]

        return tf.convert_to_tensor(kept_hyps[0].prediction, dtype=tf.int32)[None, ...]

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(self, greedy: bool = True):
        return tf.function(
            self.recognize_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.encoder.get_initial_state().get_shape(),
                              dtype=tf.float32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(),
                              dtype=tf.float32)
            ]
        )
