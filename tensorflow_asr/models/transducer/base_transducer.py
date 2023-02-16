# pylint: disable=attribute-defined-outside-init,too-many-lines
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

from tensorflow_asr.losses.rnnt_loss import RnntLoss
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.layers.base_layer import Layer
from tensorflow_asr.models.layers.embedding import Embedding
from tensorflow_asr.models.layers.one_hot_blank import OneHotBlank
from tensorflow_asr.utils import data_util, layer_util, math_util, shape_util

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))

JOINT_MODES = ["add", "mul"]


class TransducerPrediction(Layer):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        label_encoder_mode: str = "embedding",  # either "embedding" | "one_hot"
        embed_dim: int = 0,
        num_rnns: int = 1,
        rnn_units: int = 512,
        rnn_type: str = "lstm",
        rnn_implementation: int = 2,
        rnn_unroll: bool = False,
        layer_norm: bool = True,
        projection_units: int = 0,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="transducer_prediction",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if label_encoder_mode not in ["one_hot", "embedding"]:
            raise ValueError("label_encode_mode must be either 'one_hot' or 'embedding'")
        self.label_encoder_mode = label_encoder_mode
        if self.label_encoder_mode == "embedding":
            self.label_encoder = Embedding(vocab_size, embed_dim, regularizer=kernel_regularizer, name=self.label_encoder_mode)
        else:
            self.label_encoder = OneHotBlank(blank=blank, depth=vocab_size, name=self.label_encoder_mode)
        # Initialize rnn layers
        RnnClass = layer_util.get_rnn(rnn_type)
        self.rnns = []
        self.lns = []
        self.projections = []
        for i in range(num_rnns):
            rnn = RnnClass(
                units=rnn_units,
                return_sequences=True,
                name=f"{rnn_type}_{i}",
                return_state=True,
                implementation=rnn_implementation,
                unroll=rnn_unroll,
                zero_output_for_mask=True,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=tf.float32 if tf.keras.mixed_precision.global_policy().name == "mixed_bfloat16" else None,
            )
            ln = (
                tf.keras.layers.LayerNormalization(name=f"ln_{i}", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
                if layer_norm
                else None
            )
            projection = (
                tf.keras.layers.Dense(
                    projection_units,
                    name=f"projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                )
                if projection_units > 0
                else None
            )
            self.rnns.append(rnn)
            self.lns.append(ln)
            self.projections.append(projection)

    def get_initial_state(self):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(tf.stack(rnn.get_initial_state(tf.zeros([1, 1, 1], dtype=tf.float32)), axis=0))
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False):
        # inputs has shape [B, U]
        # use tf.gather_nd instead of tf.gather for tflite conversion
        outputs, prediction_length = inputs
        outputs = self.label_encoder(outputs, training=training)
        outputs = math_util.apply_mask(outputs, mask=tf.sequence_mask(prediction_length, maxlen=tf.shape(outputs)[1], dtype=tf.bool))
        for i, rnn in enumerate(self.rnns):
            orig_dtype = outputs.dtype
            if orig_dtype == tf.bfloat16:
                outputs = tf.cast(outputs, tf.float32)
            outputs = rnn(outputs, training=training, mask=getattr(outputs, "_keras_mask", None))
            outputs = outputs[0]
            if orig_dtype == tf.bfloat16:
                outputs = tf.cast(outputs, orig_dtype)
            if self.lns[i] is not None:
                outputs = self.lns[i](outputs, training=training)
            if self.projections[i] is not None:
                outputs = self.projections[i](outputs, training=training)
        return outputs

    def recognize(self, inputs, states, tflite: bool = False):
        """Recognize function for prediction network

        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]

        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        if tflite and self.label_encoder_mode == "embedding":
            outputs = self.label_encoder.recognize_tflite(inputs)
        else:
            outputs = self.label_encoder(inputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn(outputs, training=False, initial_state=tf.unstack(states[i], axis=0))
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if self.lns[i] is not None:
                outputs = self.lns[i](outputs, training=False)
            if self.projections[i] is not None:
                outputs = self.projections[i](outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)

    def compute_output_shape(self, input_shape):
        predictions_shape, _ = input_shape
        output_size = self.projections[-1].units if self.projections[-1] is not None else self.rnns[-1].units
        outputs_shape = predictions_shape + (output_size,)
        return tuple(outputs_shape)


class TransducerJointMerge(Layer):
    def __init__(self, joint_mode: str = "add", name="transducer_joint_merge", **kwargs):
        super().__init__(name=name, **kwargs)
        if joint_mode not in JOINT_MODES:
            raise ValueError(f"joint_mode must in {JOINT_MODES}")
        self.joint_mode = joint_mode

    def compute_mask(self, inputs, mask=None):
        enc_out, pred_out = inputs
        enc_mask = getattr(enc_out, "_keras_mask", None)  # BT
        pred_mask = getattr(pred_out, "_keras_mask", None)  # BU
        auto_mask = None
        if enc_mask is not None:
            auto_mask = enc_mask[:, :, tf.newaxis]  # BT1
        if pred_mask is not None:
            if auto_mask is not None:
                auto_mask = auto_mask & pred_mask[:, tf.newaxis, :]  # BT1 & B1U -> BTU
            else:
                auto_mask = pred_mask[:, tf.newaxis, :]
        if mask is not None and auto_mask is not None:
            auto_mask = auto_mask & mask
        mask = auto_mask
        return mask

    def call(self, inputs):
        enc_out, pred_out = inputs
        enc_out = tf.expand_dims(enc_out, axis=2)  # [B, T, 1, V]
        pred_out = tf.expand_dims(pred_out, axis=1)  # [B, 1, U, V]
        if self.joint_mode == "add":
            outputs = tf.add(enc_out, pred_out)  # broadcast operator
        else:
            outputs = tf.multiply(enc_out, pred_out)  # broadcast operator
        outputs = math_util.apply_mask(outputs, mask=self.compute_mask(inputs))
        return outputs  # [B, T, U, V]

    def compute_output_shape(self, input_shape):
        enc_shape, pred_shape = input_shape
        return (enc_shape[0], enc_shape[1], pred_shape[1], enc_shape[-1])


class TransducerJoint(Layer):
    def __init__(
        self,
        vocab_size: int,
        joint_dim: int = 1024,
        activation: str = "tanh",
        prejoint_encoder_linear: bool = True,
        prejoint_prediction_linear: bool = True,
        postjoint_linear: bool = False,
        joint_mode: str = "add",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="tranducer_joint",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.prejoint_encoder_linear = prejoint_encoder_linear
        self.prejoint_prediction_linear = prejoint_prediction_linear
        self.postjoint_linear = postjoint_linear

        if self.prejoint_encoder_linear:
            self.ffn_enc = tf.keras.layers.Dense(joint_dim, name="enc", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
        if self.prejoint_prediction_linear:
            self.ffn_pred = tf.keras.layers.Dense(joint_dim, use_bias=False, name="pred", kernel_regularizer=kernel_regularizer)

        self.joint = TransducerJointMerge(joint_mode=joint_mode, name="merge")

        activation = activation.lower()
        self.activation = tf.keras.layers.Activation(activation, name=activation)

        if self.postjoint_linear:
            self.ffn = tf.keras.layers.Dense(joint_dim, name="ffn", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

        self.ffn_out = tf.keras.layers.Dense(vocab_size, name="vocab", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

    def call(self, inputs, training=False):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        if self.prejoint_encoder_linear:
            enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
        if self.prejoint_prediction_linear:
            pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        outputs = self.joint([enc_out, pred_out])  # => [B, T, U, V]
        if self.postjoint_linear:
            outputs = self.ffn(outputs, training=training)
        outputs = self.activation(outputs, training=training)
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        encoder_shape, prediction_shape = input_shape
        batch_shape = encoder_shape[0]
        encoder_time_shape, prediction_time_shape = encoder_shape[1], prediction_shape[1]
        return (batch_shape, encoder_time_shape, prediction_time_shape, self.ffn_out.units)


class Transducer(BaseModel):
    """Transducer Model Warper"""

    def __init__(
        self,
        encoder: tf.keras.layers.Layer,
        blank: int,
        vocab_size: int,
        prediction_label_encoder_mode: str = "embedding",
        prediction_embed_dim: int = 512,
        prediction_num_rnns: int = 1,
        prediction_rnn_units: int = 320,
        prediction_rnn_type: str = "lstm",
        prediction_rnn_implementation: int = 2,
        prediction_rnn_unroll: bool = False,
        prediction_layer_norm: bool = True,
        prediction_projection_units: int = 0,
        prediction_trainable: bool = True,
        joint_dim: int = 1024,
        joint_activation: str = "tanh",
        joint_mode: str = "add",
        joint_trainable: bool = True,
        prejoint_encoder_linear: bool = True,
        prejoint_prediction_linear: bool = True,
        postjoint_linear: bool = False,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="transducer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            blank=blank,
            vocab_size=vocab_size,
            label_encoder_mode=prediction_label_encoder_mode,
            embed_dim=prediction_embed_dim,
            num_rnns=prediction_num_rnns,
            rnn_units=prediction_rnn_units,
            rnn_type=prediction_rnn_type,
            rnn_implementation=prediction_rnn_implementation,
            rnn_unroll=prediction_rnn_unroll,
            layer_norm=prediction_layer_norm,
            projection_units=prediction_projection_units,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=prediction_trainable,
            name="prediction",
        )
        self.joint_net = TransducerJoint(
            vocab_size=vocab_size,
            joint_dim=joint_dim,
            activation=joint_activation,
            prejoint_encoder_linear=prejoint_encoder_linear,
            prejoint_prediction_linear=prejoint_prediction_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=joint_trainable,
            name="joint",
        )
        self.time_reduction_factor = 1
        self.decoder_gwn_step = None
        self.decoder_gwn_stddev = None

    def make(self, input_shape, prediction_shape=[None], batch_size=None):
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

    def compile(
        self,
        optimizer,
        blank=0,
        run_eagerly=None,
        mxp="none",
        ga_steps=None,
        apply_gwn_config=None,
        **kwargs,
    ):
        loss = RnntLoss(blank=blank)
        super().compile(
            loss=loss,
            optimizer=optimizer,
            run_eagerly=run_eagerly,
            mxp=mxp,
            ga_steps=ga_steps,
            apply_gwn_config=apply_gwn_config,
            **kwargs,
        )

    def apply_gwn(self):
        if self.apply_gwn_config:
            original_weights = {}
            if self.apply_gwn_config.get("encoder_step") and self.apply_gwn_config.get("encoder_stddev"):
                original_weights["encoder"] = tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.apply_gwn_config["encoder_step"]),
                    lambda: layer_util.add_gwn(self.encoder.trainable_weights, stddev=self.apply_gwn_config["encoder_stddev"]),
                    lambda: self.encoder.trainable_weights,
                )
            if self.apply_gwn_config.get("predict_net_step") and self.apply_gwn_config.get("predict_net_stddev"):
                original_weights["predict_net"] = tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.apply_gwn_config["predict_net_step"]),
                    lambda: layer_util.add_gwn(self.predict_net.trainable_weights, stddev=self.apply_gwn_config["predict_net_stddev"]),
                    lambda: self.predict_net.trainable_weights,
                )
            if self.apply_gwn_config.get("joint_net_step") and self.apply_gwn_config.get("joint_net_stddev"):
                original_weights["joint_net"] = tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.apply_gwn_config["joint_net_step"]),
                    lambda: layer_util.add_gwn(self.joint_net.trainable_weights, stddev=self.apply_gwn_config["joint_net_stddev"]),
                    lambda: self.joint_net.trainable_weights,
                )
            return original_weights
        return []

    def remove_gwn(self, original_weights):
        if self.apply_gwn_config:
            if original_weights.get("encoder") is not None:
                tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.apply_gwn_config["encoder_step"]),
                    lambda: layer_util.sub_gwn(original_weights["encoder"], self.encoder.trainable_weights),
                    lambda: None,
                )
            if original_weights.get("predict_net") is not None:
                tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.apply_gwn_config["predict_net_step"]),
                    lambda: layer_util.sub_gwn(original_weights["predict_net"], self.predict_net.trainable_weights),
                    lambda: None,
                )
            if original_weights.get("joint_net") is not None:
                tf.cond(
                    tf.greater_equal((self.optimizer.iterations), self.apply_gwn_config["joint_net_step"]),
                    lambda: layer_util.sub_gwn(original_weights["joint_net"], self.joint_net.trainable_weights),
                    lambda: None,
                )

    def call(self, inputs, training=False):
        enc, enc_length = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=training)
        pred = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=training)
        logits = self.joint_net([enc, pred], training=training)
        return data_util.create_logits(logits=logits, logits_length=enc_length)

    # -------------------------------- INFERENCES -------------------------------------

    def preprocess(self, signals: tf.Tensor):
        with tf.name_scope("preprocess"):
            batch = tf.constant(0, dtype=tf.int32)
            total_batch = tf.shape(signals)[0]

            inputs = tf.TensorArray(
                dtype=tf.float32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape(self.speech_featurizer.shape),
            )

            inputs_length = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([]),
            )

            def condition(_batch, _inputs, _inputs_length):
                return tf.less(_batch, total_batch)

            def body(_batch, _inputs, _inputs_length):
                item_inputs = self.speech_featurizer.tf_extract(signals[_batch])
                item_inputs_length = tf.cast(tf.shape(item_inputs)[0], tf.int32)
                _inputs = _inputs.write(_batch, item_inputs)
                _inputs_length = _inputs_length.write(_batch, item_inputs_length)
                return _batch + 1, _inputs, _inputs_length

            batch, inputs, inputs_length = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, inputs, inputs_length],
            )
            inputs = math_util.pad_tfarray(inputs, blank=0.0, element_axis=0)

            return inputs.stack(), inputs_length.stack()

    def encoder_inference(self, features: tf.Tensor):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope("encoder"):
            inputs_length = tf.expand_dims(tf.shape(features)[0], axis=0)
            outputs = tf.expand_dims(features, axis=0)
            outputs, inputs_length = self.encoder([outputs, inputs_length], training=False)
            return tf.squeeze(outputs, axis=0)

    def decoder_inference(self, encoded: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor, tflite: bool = False):
        """Infer function for decoder

        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope("decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predict_net.recognize(predicted, states, tflite=tflite)  # [1, 1, P], states
            ytu = tf.nn.log_softmax(self.joint_net([encoded, y], training=False))  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    # -------------------------------- GREEDY -------------------------------------

    def recognize(self, inputs: Dict[str, tf.Tensor]):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of extracted features
            input_length (tf.Tensor): a batch of extracted features length

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded, encoded_length = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=False)
        return self._perform_greedy_batch(encoded=encoded, encoded_length=encoded_length)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def recognize_from_signal(self, signals: tf.Tensor):
        """
        RNN Transuder Greedy Decoding From Batch of Signals

        Args:
            signals (tf.Tensor): batch of signals in shape [B, None]

        Returns:
            tf.Tensor: batch of decoded transcripts in shape [B]
        """
        inputs, inputs_length = self.preprocess(signals)
        return self.recognize(data_util.create_inputs(inputs=inputs, inputs_length=inputs_length))

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
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, hypothesis.states

    def recognize_tflite_with_timestamp(self, signal, predicted, states):
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
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

    def _perform_greedy_batch(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        with tf.name_scope("perform_greedy_batch"):
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
                hypothesis = self._perform_greedy_v2(
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
                swap_memory=swap_memory,
            )

            decoded = math_util.pad_tfarray(decoded, blank=self.text_featurizer.blank)
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
        with tf.name_scope("greedy"):
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
        with tf.name_scope("greedy_v2"):
            time = tf.constant(0, dtype=tf.int32)
            pred_index = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=(2 * total),
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _pred_index, _hypothesis):
                return tf.logical_and(tf.less(_time, total), tf.less(_pred_index, 2 * total - 1))

            def body(_time, _pred_index, _hypothesis):
                ytu, _states = self.decoder_inference(
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),  # avoid using [index] in tflite
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                _equal_blank = tf.equal(_predict, self.text_featurizer.blank)
                _time = tf.where(_equal_blank, _time + 1, _time)
                _index = tf.where(_equal_blank, _hypothesis.index, _predict)
                _states = tf.where(_equal_blank, _hypothesis.states, _states)
                _pred_index = tf.where(_equal_blank, _pred_index, _pred_index + 1)
                _prediction = _hypothesis.prediction.write(_pred_index, _index)

                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time, _pred_index, _hypothesis

            time, pred_index, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, pred_index, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(self, inputs: Dict[str, tf.Tensor], lm: bool = False):
        """
        RNN Transducer Beam Search
        Args:
            inputs (Dict[str, tf.Tensor]): Input dictionary containing "inputs" and "inputs_length"
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded, encoded_length = self.encoder([inputs["inputs"], inputs["inputs_length"]], training=False)
        return self._perform_beam_search_batch(encoded=encoded, encoded_length=encoded_length, lm=lm)

    def _perform_beam_search_batch(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        lm: bool = False,
        parallel_iterations: int = 10,
        swap_memory: bool = True,
    ):
        with tf.name_scope("perform_beam_search_batch"):
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

            decoded = math_util.pad_tfarray(decoded, blank=self.text_featurizer.blank)
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
        with tf.name_scope("beam_search"):
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
                    prediction=A.prediction.unstack(math_util.pad_tfarray(B.prediction, blank=self.text_featurizer.blank).stack()),
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
                        math_util.pad_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
                        y_hat_score_index,
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
                                math_util.pad_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
                                remain_indices,
                            )
                        ),
                        states=A.states.unstack(tf.gather_nd(A.states.stack(), remain_indices)),
                    )
                    A_i = tf.cond(tf.equal(A_i, 0), true_fn=lambda: A_i, false_fn=lambda: A_i - 1)

                    ytu, new_states = self.decoder_inference(encoded=encoded_t, predicted=y_hat_index, states=y_hat_states, tflite=tflite)

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
                            scatter_index = math_util.count_non_blank(y_hat_prediction, blank=self.text_featurizer.blank)
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
            prediction = math_util.pad_tfarray(B.prediction, blank=self.text_featurizer.blank).stack()
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
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32),
            ],
        )
