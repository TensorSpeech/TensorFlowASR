# pylint: disable=attribute-defined-outside-init,too-many-lines
# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import schemas
from tensorflow_asr.losses.rnnt_loss import RnntLoss
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.layers.embedding import Embedding, OneHotBlank
from tensorflow_asr.utils import layer_util, shape_util

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
        assert label_encoder_mode in ("one_hot", "embedding"), "label_encode_mode must be either 'one_hot' or 'embedding'"
        self.label_encoder = (
            Embedding(vocab_size, embed_dim, regularizer=kernel_regularizer, name=label_encoder_mode, dtype=self.dtype)
            if label_encoder_mode == "embedding"
            else OneHotBlank(blank=blank, depth=vocab_size, name=label_encoder_mode, dtype=self.dtype)
        )
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
                dtype=self.dtype,
            )
            ln = (
                tf.keras.layers.LayerNormalization(
                    name=f"ln_{i}", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
                )
                if layer_norm
                else None
            )
            projection = (
                tf.keras.layers.Dense(
                    projection_units,
                    name=f"projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    dtype=self.dtype,
                )
                if projection_units > 0
                else None
            )
            self.rnns.append(rnn)
            self.lns.append(ln)
            self.projections.append(projection)

    def get_initial_state(self, batch_size: int):
        """
        Get zeros states

        Returns
        -------
        tf.Tensor, shape [B, num_rnns, nstates, state_size]
            Zero initialized states
        """
        states = []
        for rnn in self.rnns:
            states.append(tf.stack(rnn.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype)), axis=0))
        return tf.transpose(tf.stack(states, axis=0), perm=[2, 0, 1, 3])

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.label_encoder((outputs, outputs_length), training=training)
        for i, rnn in enumerate(self.rnns):
            outputs, *_ = rnn(outputs, training=training)  # mask auto populate
            if self.lns[i] is not None:
                outputs = self.lns[i](outputs, training=training)
            if self.projections[i] is not None:
                outputs = self.projections[i](outputs, training=training)
        return outputs, outputs_length

    def call_next(self, inputs, previous_decoder_states):
        """
        Recognize function for prediction network from the previous predicted tokens

        Parameters
        ----------
        inputs : tf.Tensor, shape [B, 1]
        previous_decoder_states : tf.Tensor, shape [B, num_rnns, nstates, rnn_units]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor], shapes ([B, 1, rnn_units], [B, num_rnns, nstates, rnn_units])
            Outputs, new states
        """
        with tf.name_scope(f"{self.name}_call_next"):
            previous_decoder_states = tf.transpose(previous_decoder_states, perm=[1, 2, 0, 3])
            outputs = self.label_encoder.call_next(inputs)
            new_states = []
            for i, rnn in enumerate(self.rnns):
                outputs, *_states = rnn(outputs, training=False, initial_state=tf.unstack(previous_decoder_states[i], axis=0))
                new_states.append(tf.stack(_states))
                if self.lns[i] is not None:
                    outputs = self.lns[i](outputs, training=False)
                if self.projections[i] is not None:
                    outputs = self.projections[i](outputs, training=False)
            return outputs, tf.transpose(tf.stack(new_states, axis=0), perm=[2, 0, 1, 3])

    def compute_mask(self, inputs, mask=None):
        return self.label_encoder.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape, output_length_shape = self.label_encoder.compute_output_shape((output_shape, output_length_shape))
        for i, rnn in enumerate(self.rnns):
            output_shape = (
                self.projections[i].compute_output_shape(output_shape)
                if self.projections[i] is not None
                else rnn.compute_output_shape(output_shape)[0]
            )
        return tuple(output_shape), tuple(output_length_shape)


class TransducerJointMerge(Layer):
    def __init__(self, joint_mode: str = "add", name="transducer_joint_merge", **kwargs):
        super().__init__(name=name, **kwargs)
        if joint_mode not in JOINT_MODES:
            raise ValueError(f"joint_mode must in {JOINT_MODES}")
        self.joint_mode = joint_mode

    def compute_mask(self, inputs, mask=None):
        enc_out, pred_out = inputs
        enc_mask = mask[0] if mask else getattr(enc_out, "_keras_mask", None)  # BT
        pred_mask = mask[1] if mask else getattr(pred_out, "_keras_mask", None)  # BU
        auto_mask = None
        if enc_mask is not None:
            auto_mask = enc_mask[:, :, tf.newaxis]  # BT1
        if pred_mask is not None:
            if auto_mask is not None:
                auto_mask = auto_mask & pred_mask[:, tf.newaxis, :]  # BT1 & B1U -> BTU
            else:
                auto_mask = pred_mask[:, tf.newaxis, :]
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
        return outputs  # [B, T, U, V]

    def compute_output_shape(self, input_shape):
        enc_shape, pred_shape = input_shape
        return enc_shape[0], enc_shape[1], pred_shape[1], enc_shape[-1]


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
            self.ffn_enc = tf.keras.layers.Dense(
                joint_dim,
                name="enc",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=self.dtype,
            )
        if self.prejoint_prediction_linear:
            self.ffn_pred = tf.keras.layers.Dense(
                joint_dim,
                use_bias=False,
                name="pred",
                kernel_regularizer=kernel_regularizer,
                dtype=self.dtype,
            )

        self.joint = TransducerJointMerge(joint_mode=joint_mode, name="merge", dtype=self.dtype)

        activation = activation.lower()
        self.activation = tf.keras.layers.Activation(activation, name=activation, dtype=self.dtype)

        if self.postjoint_linear:
            self.ffn = tf.keras.layers.Dense(
                joint_dim,
                name="ffn",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=self.dtype,
            )

        self.ffn_out = tf.keras.layers.Dense(
            vocab_size,
            name="vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        if self.prejoint_encoder_linear:
            enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
        if self.prejoint_prediction_linear:
            pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        outputs = self.joint((enc_out, pred_out))  # => [B, T, U, V]
        if self.postjoint_linear:
            outputs = self.ffn(outputs, training=training)
        outputs = self.activation(outputs, training=training)
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self.joint.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        encoder_shape, prediction_shape = input_shape
        batch_shape = encoder_shape[0]
        encoder_time_shape, prediction_time_shape = encoder_shape[1], prediction_shape[1]
        return batch_shape, encoder_time_shape, prediction_time_shape, self.ffn_out.units


class Transducer(BaseModel):
    """Transducer Model Warper"""

    def __init__(
        self,
        blank: int,
        vocab_size: int,
        speech_config: dict,
        encoder: tf.keras.layers.Layer,
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
        super().__init__(speech_config=speech_config, name=name, **kwargs)
        self.blank = blank
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
            dtype=self.dtype,
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
            dtype=self.dtype,
        )
        self.time_reduction_factor = 1

    def compile(self, optimizer, output_shapes=None, **kwargs):
        loss = RnntLoss(blank=self.blank, output_shapes=output_shapes, name="rnnt_loss")
        return super().compile(loss, optimizer, **kwargs)

    def apply_gwn(self):
        if self.gwn_config:
            original_weights = {}
            if self.gwn_config.get("encoder_step") is not None and self.gwn_config.get("encoder_stddev") is not None:
                original_weights["encoder"] = tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["encoder_step"]),
                    lambda: layer_util.add_gwn(self.encoder.trainable_weights, stddev=self.gwn_config["encoder_stddev"]),
                    lambda: self.encoder.trainable_weights,
                )
            if self.gwn_config.get("predict_net_step") is not None and self.gwn_config.get("predict_net_stddev") is not None:
                original_weights["predict_net"] = tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["predict_net_step"]),
                    lambda: layer_util.add_gwn(self.predict_net.trainable_weights, stddev=self.gwn_config["predict_net_stddev"]),
                    lambda: self.predict_net.trainable_weights,
                )
            if self.gwn_config.get("joint_net_step") is not None and self.gwn_config.get("joint_net_stddev") is not None:
                original_weights["joint_net"] = tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["joint_net_step"]),
                    lambda: layer_util.add_gwn(self.joint_net.trainable_weights, stddev=self.gwn_config["joint_net_stddev"]),
                    lambda: self.joint_net.trainable_weights,
                )
            return original_weights
        return {}

    def remove_gwn(self, original_weights):
        if self.gwn_config:
            if original_weights.get("encoder") is not None:
                tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["encoder_step"]),
                    lambda: layer_util.sub_gwn(original_weights["encoder"], self.encoder.trainable_weights),
                    lambda: None,
                )
            if original_weights.get("predict_net") is not None:
                tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["predict_net_step"]),
                    lambda: layer_util.sub_gwn(original_weights["predict_net"], self.predict_net.trainable_weights),
                    lambda: None,
                )
            if original_weights.get("joint_net") is not None:
                tf.cond(
                    tf.greater_equal(self.optimizer.iterations, self.gwn_config["joint_net_step"]),
                    lambda: layer_util.sub_gwn(original_weights["joint_net"], self.joint_net.trainable_weights),
                    lambda: None,
                )

    def call(self, inputs: schemas.TrainInput, training=False):
        features, features_length = self.feature_extraction((inputs["inputs"], inputs["inputs_length"]), training=training)
        enc, logits_length, caching = self.encoder((features, features_length, inputs.get("caching")), training=training)
        pred, _ = self.predict_net((inputs["predictions"], inputs["predictions_length"]), training=training)
        logits = self.joint_net((enc, pred), training=training)
        return schemas.TrainOutput(
            logits=logits,
            logits_length=logits_length,
            caching=caching,
        )

    def call_next(
        self,
        current_frames: tf.Tensor,
        previous_tokens: tf.Tensor,
        previous_decoder_states: tf.Tensor,
    ):
        """
        Decode current frame given previous predicted token and states

        Parameters
        ----------
        current_frames : tf.Tensor, shape [B, 1, E]
            Output of the encoder network of the current frame
        previous_tokens : tf.Tensor, shape [B, 1]
            Predicted token of the previous frame
        previous_decoder_states : tf.Tensor, shape [B, num_rnns, nstates, state_size]
            States got from previous frame

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor], shapes ([B, 1, 1, V], [B, num_rnns, nstates, state_size])
            Output of joint network of the current frame, new states of prediction network
        """
        with tf.name_scope(f"{self.name}_call_next"):
            y, new_states = self.predict_net.call_next(previous_tokens, previous_decoder_states)
            ytu = self.joint_net([current_frames, y], training=False)
            ytu = tf.nn.log_softmax(ytu)
            return ytu, new_states

    def get_initial_tokens(self, batch_size=1):
        return super().get_initial_tokens(batch_size)

    def get_initial_encoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    def get_initial_decoder_states(self, batch_size=1):
        return self.predict_net.get_initial_state(batch_size)

    # -------------------------------- GREEDY -------------------------------------

    def recognize(self, inputs: schemas.PredictInput, max_tokens_per_frame: int = 3, **kwargs):
        """
        Recognize greedy from input signals

        Parameters
        ----------
        inputs : schemas.PredictInput

        Returns
        -------
        named tuple of
            (
                tokens, will be feed to text_featurizer.detokenize or text_featurizer.detokenize_unicode_points,
                next_encoder_states, if encoder does not have states, returns None, will be used to predict next chunk of audio,
                next_tokens, will be used to predict next chunk of audio,
                next_decoder_states, next states of predict_net, will be used to predict next chunk of audio,
            )
        """
        if self._batch_size == 1:
            return self.recognize_single(inputs, max_tokens_per_frame=max_tokens_per_frame, **kwargs)
        return self.recognize_batch(inputs, **kwargs)

    def recognize_batch(self, inputs: schemas.PredictInput, **kwargs):
        """
        Ref: https://arxiv.org/pdf/1801.00841.pdf
        This is a greedy decoding algorithm that greedily select the best token at each time step
        Only apply for batch size > 1
        """
        with tf.name_scope(f"{self.name}_recognize"):
            features, features_length = self.feature_extraction((inputs.inputs, inputs.inputs_length), training=False)
            encoded, encoded_length, next_encoder_states = self.encoder.call_next(features, features_length, inputs.previous_encoder_states)

            nframes = tf.expand_dims(encoded_length, axis=-1)  # [B, 1]
            batch_size, max_frames, _ = shape_util.shape_list(encoded)
            # The current indices of the output of encoder, shape [B, 1]
            frame_indices = tf.zeros([batch_size, 1], dtype=tf.int32, name="frame_indices")
            # Previous predicted tokens, initially are blanks, shape [B, 1]
            previous_tokens = inputs.previous_tokens
            # Previous states of the prediction network, initially are zeros, shape [B, num_rnns, nstates, rnn_units]
            previous_decoder_states = inputs.previous_decoder_states
            # Assumption that number of tokens can not exceed (2 * the size of output of encoder + 1), this is for static runs like TPU or TFLite
            max_tokens = max_frames * 2 + 1
            # All of the tokens that are getting recognized, initially are blanks, shape [B, nframes * 2 + 1]
            tokens = tf.ones([batch_size, max_tokens], dtype=tf.int32, name="tokens") * self.blank
            # The current indices of the token that are currently being recognized, shape [B, 1], the tokens indices are started with 1 so that any
            # blank token recognized got updated to index 0 to avoid affecting results
            tokens_indices = tf.ones([batch_size, 1], dtype=tf.int32, name="tokens_indices")

            def cond(_frame_indices, _previous_tokens, _previous_decoder_states, _tokens, _tokens_indices):
                return tf.logical_not(  # Reversed so that the loop check and continue
                    # One of the following condition met will terminate the loop
                    tf.logical_or(
                        # Stop when ALL of the indices of the output of the encoder reach the end
                        tf.math.reduce_all(tf.greater_equal(_frame_indices, nframes - 1)),
                        # Stop when ALL of the indices of recognized tokens reach the end
                        tf.math.reduce_all(tf.greater_equal(_tokens_indices, max_tokens - 1)),
                    )
                )

            def body(_frame_indices, _previous_tokens, _previous_decoder_states, _tokens, _tokens_indices):
                _current_frames = tf.expand_dims(tf.gather_nd(encoded, tf.minimum(_frame_indices, nframes - 1), batch_dims=1), axis=1)  # [B, 1, E]
                _log_softmax, _states = self.call_next(_current_frames, _previous_tokens, _previous_decoder_states)
                _current_tokens = tf.reshape(tf.argmax(_log_softmax, axis=-1, output_type=tf.int32), [batch_size, 1])  # [B, 1, 1] -> [B, 1]
                # conditions, blanks are ignored
                _equal_blank = tf.equal(_current_tokens, self.blank)  # [B, 1]
                # if the token index >= max tokens, it's already finished, set to blank to ignore
                _equal_blank = tf.logical_or(_equal_blank, tf.greater_equal(_tokens_indices, max_tokens))
                # if the frame index > nframes, it's already done, set to blank to ignore
                _equal_blank = tf.logical_or(_equal_blank, tf.greater(_frame_indices, nframes))
                # update results
                _update_tokens = tf.reshape(tf.where(_equal_blank, self.blank, _current_tokens), [batch_size])  # [B]
                _update_tokens_indices = tf.where(
                    _equal_blank, 0, tf.minimum(tf.add(_tokens_indices, 1), max_tokens - 1)
                )  # blanks are getting updated at index 0 to avoid affecting results
                _tokens = tf.tensor_scatter_nd_update(
                    tensor=_tokens,
                    indices=tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=-1), _update_tokens_indices], -1),  # [B, 2]
                    updates=_update_tokens,  # [B]
                )
                _tokens_indices = tf.where(_equal_blank, _tokens_indices, tf.minimum(tf.add(_tokens_indices, 1), max_tokens - 1))
                # update states
                _frame_indices = tf.where(_equal_blank, tf.add(_frame_indices, 1), _frame_indices)  # blank then next frames, else current frames
                _previous_tokens = tf.where(_equal_blank, _previous_tokens, _current_tokens)  # blank then keep prev tokens, else next tokens
                _previous_decoder_states = tf.where(
                    tf.reshape(_equal_blank, [batch_size, 1, 1, 1]), _previous_decoder_states, _states
                )  # blank then keep prev states, else next states # pylint: disable=line-too-long
                return _frame_indices, _previous_tokens, _previous_decoder_states, _tokens, _tokens_indices

            (
                frame_indices,
                next_tokens,
                next_decoder_states,
                tokens,
                tokens_indices,
            ) = tf.while_loop(cond, body, loop_vars=(frame_indices, previous_tokens, previous_decoder_states, tokens, tokens_indices))

            return schemas.PredictOutput(
                tokens=tokens,
                next_tokens=next_tokens,
                next_encoder_states=next_encoder_states,
                next_decoder_states=next_decoder_states,
            )

    def recognize_single(self, inputs: schemas.PredictInput, max_tokens_per_frame: int = 3, **kwargs):
        """
        Ref: https://arxiv.org/pdf/1801.00841.pdf
        This is a greedy decoding algorithm that greedily select the best token at each time step
        Only apply for batch size 1
        """
        with tf.name_scope(f"{self.name}_decode_greedy"):
            features, features_length = self.feature_extraction((inputs.inputs, inputs.inputs_length), training=False)
            encoded, encoded_length, next_encoder_states = self.encoder.call_next(features, features_length, inputs.previous_encoder_states)

            frame = tf.zeros([1, 1], dtype=tf.int32)
            nframes = encoded_length

            previous_tokens = inputs.previous_tokens
            token_index = tf.ones([], dtype=tf.int32) * -1
            tokens = tf.TensorArray(
                dtype=tf.int32,
                size=tf.reshape(nframes, shape=[]) * max_tokens_per_frame,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([]),
            )
            num_tokens_per_frame = tf.TensorArray(
                dtype=tf.int32,
                size=tf.reshape(nframes, shape=[]),
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([]),
            )

            previous_decoder_states = inputs.previous_decoder_states

            def condition(
                _frame,
                _nframes,
                _previous_tokens,
                _token_index,
                _tokens,
                _num_tokens_per_frame,
                _max_tokens_per_frame,
                _previous_decoder_states,
            ):
                return tf.less(_frame, _nframes)

            def body(
                _frame,
                _nframes,
                _previous_tokens,
                _token_index,
                _tokens,
                _num_tokens_per_frame,
                _max_tokens_per_frame,
                _previous_decoder_states,
            ):
                _current_frame = tf.expand_dims(tf.gather_nd(encoded, _frame, batch_dims=1), axis=1)  # [1, 1, E]
                _log_softmax, _states = self.call_next(_current_frame, _previous_tokens, _previous_decoder_states)
                _current_tokens = tf.reshape(tf.argmax(_log_softmax, axis=-1, output_type=tf.int32), [1, 1])  # [1, 1, 1] -> [1, 1]

                ##################### conditions, blanks are ignored
                _equal_blank = tf.equal(_current_tokens, self.blank)  # [1, 1]

                ##################### step updates
                __frame_index = tf.reshape(_frame, shape=[])
                __equal_blank_index = tf.reshape(_equal_blank, shape=[])
                # only non-blank tokens are counted in number of tokens per frame
                _current_frame_num_tokens = tf.where(
                    __equal_blank_index,
                    _num_tokens_per_frame.read(__frame_index),
                    tf.add(_num_tokens_per_frame.read(__frame_index), 1),
                )
                _num_tokens_per_frame = _num_tokens_per_frame.write(__frame_index, _current_frame_num_tokens)
                # increase frame index if current tokens are blank or number of tokens per frame exceeds max tokens per frame
                _frame = tf.where(
                    tf.logical_or(_equal_blank, tf.greater_equal(_current_frame_num_tokens, _max_tokens_per_frame)),
                    tf.add(_frame, 1),
                    _frame,
                )
                # increase token index if current token is not blank, so that it can be appended to tokens array
                _token_index = tf.where(__equal_blank_index, _token_index, tf.add(_token_index, 1))

                ##################### content updates
                # keep previous tokens if current tokens are blank
                _current_tokens = tf.where(_equal_blank, _previous_tokens, _current_tokens)
                # keep previous states if current tokens are blank
                _states = tf.where(tf.reshape(_equal_blank, [1, 1, 1, 1]), _previous_decoder_states, _states)
                # token_index initialized as -1, so that the first recognized token will be at index 0
                # therefore only update (append) tokens when token_index >= 0
                _tokens = tf.cond(
                    tf.greater_equal(_token_index, 0),
                    lambda: _tokens.write(_token_index, tf.reshape(_current_tokens, shape=[])),
                    lambda: _tokens,
                )

                ##################### return
                return (
                    _frame,
                    _nframes,
                    _current_tokens,
                    _token_index,
                    _tokens,
                    _num_tokens_per_frame,
                    _max_tokens_per_frame,
                    _states,
                )

            (
                frame,
                nframes,
                next_tokens,
                token_index,
                tokens,
                num_tokens_per_frame,
                max_tokens_per_frame,
                next_decoder_states,
            ) = tf.while_loop(
                condition,
                body,
                loop_vars=(
                    frame,
                    nframes,
                    previous_tokens,
                    token_index,
                    tokens,
                    num_tokens_per_frame,
                    max_tokens_per_frame,
                    previous_decoder_states,
                ),
                back_prop=False,
            )

            return schemas.PredictOutput(
                tokens=tf.reshape(tokens.stack(), shape=[1, -1]),
                next_tokens=next_tokens,
                next_encoder_states=next_encoder_states,
                next_decoder_states=next_decoder_states,
            )

    # def recognize_tflite_with_timestamp(self, signal, predicted, states):
    #     features = self.speech_featurizer.tf_extract(signal)
    #     encoded = self.encoder_inference(features)
    #     hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
    #     indices = self.text_featurizer.normalize_indices(hypothesis.prediction)
    #     upoints = tf.gather_nd(self.text_featurizer.upoints, tf.expand_dims(indices, axis=-1))  # [None, max_subword_length]

    #     num_samples = tf.cast(tf.shape(signal)[0], dtype=tf.float32)
    #     total_time_reduction_factor = self.time_reduction_factor * self.speech_featurizer.frame_step

    #     stime = tf.range(0, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
    #     stime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

    #     etime = tf.range(total_time_reduction_factor, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
    #     etime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

    #     non_blank = tf.where(tf.not_equal(upoints, 0))
    #     non_blank_transcript = tf.gather_nd(upoints, non_blank)
    #     non_blank_stime = tf.gather_nd(tf.repeat(tf.expand_dims(stime, axis=-1), tf.shape(upoints)[-1], axis=-1), non_blank)
    #     non_blank_etime = tf.gather_nd(tf.repeat(tf.expand_dims(etime, axis=-1), tf.shape(upoints)[-1], axis=-1), non_blank)

    #     return non_blank_transcript, non_blank_stime, non_blank_etime, hypothesis.index, hypothesis.states

    # def _perform_greedy_batch(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = False,
    # ):
    #     with tf.name_scope("perform_greedy_batch"):
    #         total_batch = tf.shape(encoded)[0]
    #         batch = tf.constant(0, dtype=tf.int32)

    #         decoded = tf.TensorArray(
    #             dtype=tf.int32,
    #             size=total_batch,
    #             dynamic_size=False,
    #             clear_after_read=False,
    #             element_shape=tf.TensorShape([None]),
    #         )

    #         def condition(batch, _):
    #             return tf.less(batch, total_batch)

    #         def body(batch, decoded):
    #             hypothesis = self._perform_greedy_v2(
    #                 encoded=encoded[batch],
    #                 encoded_length=encoded_length[batch],
    #                 predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
    #                 states=self.predict_net.get_initial_state(),
    #                 parallel_iterations=parallel_iterations,
    #                 swap_memory=swap_memory,
    #             )
    #             decoded = decoded.write(batch, hypothesis.prediction)
    #             return batch + 1, decoded

    #         batch, decoded = tf.while_loop(
    #             condition,
    #             body,
    #             loop_vars=[batch, decoded],
    #             parallel_iterations=parallel_iterations,
    #             swap_memory=swap_memory,
    #         )

    #         decoded = math_util.pad_tfarray(decoded, blank=self.text_featurizer.blank)
    #         return self.text_featurizer.detokenize(decoded.stack())

    # def _perform_greedy(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     predicted: tf.Tensor,
    #     states: tf.Tensor,
    #     tflite: bool = False,
    # ):
    #     """Ref: https://arxiv.org/pdf/1801.00841.pdf"""
    #     with tf.name_scope("greedy_v2"):
    #         time = tf.constant(0, dtype=tf.int32)
    #         pred_index = tf.constant(0, dtype=tf.int32)
    #         total = encoded_length

    #         hypothesis = Hypothesis(
    #             index=predicted,
    #             prediction=tf.TensorArray(
    #                 dtype=tf.int32,
    #                 size=(2 * total),
    #                 dynamic_size=False,
    #                 clear_after_read=False,
    #                 element_shape=tf.TensorShape([]),
    #             ),
    #             states=states,
    #         )

    #         def condition(_time, _pred_index, _hypothesis):
    #             return tf.logical_and(tf.less(_time, total), tf.less(_pred_index, 2 * total - 1))

    #         def body(_time, _pred_index, _hypothesis):
    #             ytu, _states = self.decoder_inference(
    #                 encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),  # avoid using [index] in tflite
    #                 predicted=_hypothesis.index,
    #                 states=_hypothesis.states,
    #                 tflite=tflite,
    #             )
    #             _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

    #             _equal_blank = tf.equal(_predict, self.text_featurizer.blank)
    #             _time = tf.where(_equal_blank, _time + 1, _time)
    #             _index = tf.where(_equal_blank, _hypothesis.index, _predict)
    #             _states = tf.where(_equal_blank, _hypothesis.states, _states)
    #             _pred_index = tf.where(_equal_blank, _pred_index, _pred_index + 1)
    #             _prediction = _hypothesis.prediction.write(_pred_index, _index)

    #             _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

    #             return _time, _pred_index, _hypothesis

    #         time, pred_index, hypothesis = tf.while_loop(condition, body, loop_vars=[time, pred_index, hypothesis])

    #         return Hypothesis(
    #             index=hypothesis.index,
    #             prediction=hypothesis.prediction.stack(),
    #             states=hypothesis.states,
    #         )

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(self, inputs: schemas.PredictInput, beam_width: int = 10, **kwargs):
        return self.recognize(inputs=inputs, **kwargs)  # TODO: Implement beam search

    # def _perform_beam_search_batch(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     lm: bool = False,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = True,
    # ):
    #     with tf.name_scope("perform_beam_search_batch"):
    #         total_batch = tf.shape(encoded)[0]
    #         batch = tf.constant(0, dtype=tf.int32)

    #         decoded = tf.TensorArray(
    #             dtype=tf.int32,
    #             size=total_batch,
    #             dynamic_size=False,
    #             clear_after_read=False,
    #             element_shape=None,
    #         )

    #         def condition(batch, _):
    #             return tf.less(batch, total_batch)

    #         def body(batch, decoded):
    #             hypothesis = self._perform_beam_search(
    #                 encoded[batch],
    #                 encoded_length[batch],
    #                 lm,
    #                 parallel_iterations=parallel_iterations,
    #                 swap_memory=swap_memory,
    #             )
    #             decoded = decoded.write(batch, hypothesis.prediction)
    #             return batch + 1, decoded

    #         batch, decoded = tf.while_loop(
    #             condition,
    #             body,
    #             loop_vars=[batch, decoded],
    #             parallel_iterations=parallel_iterations,
    #             swap_memory=True,
    #         )

    #         decoded = math_util.pad_tfarray(decoded, blank=self.text_featurizer.blank)
    #         return self.text_featurizer.detokenize(decoded.stack())

    # def _perform_beam_search(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     lm: bool = False,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = True,
    #     tflite: bool = False,
    # ):
    #     with tf.name_scope("beam_search"):
    #         beam_width = tf.where(
    #             tf.less(self.text_featurizer.decoder_config.beam_width, self.text_featurizer.num_classes),
    #             self.text_featurizer.decoder_config.beam_width,
    #             self.text_featurizer.num_classes - 1,
    #         )
    #         total = encoded_length

    #         def initialize_beam(dynamic=False):
    #             return BeamHypothesis(
    #                 score=tf.TensorArray(
    #                     dtype=tf.float32,
    #                     size=beam_width if not dynamic else 0,
    #                     dynamic_size=dynamic,
    #                     element_shape=tf.TensorShape([]),
    #                     clear_after_read=False,
    #                 ),
    #                 indices=tf.TensorArray(
    #                     dtype=tf.int32,
    #                     size=beam_width if not dynamic else 0,
    #                     dynamic_size=dynamic,
    #                     element_shape=tf.TensorShape([]),
    #                     clear_after_read=False,
    #                 ),
    #                 prediction=tf.TensorArray(
    #                     dtype=tf.int32,
    #                     size=beam_width if not dynamic else 0,
    #                     dynamic_size=dynamic,
    #                     element_shape=None,
    #                     clear_after_read=False,
    #                 ),
    #                 states=tf.TensorArray(
    #                     dtype=tf.float32,
    #                     size=beam_width if not dynamic else 0,
    #                     dynamic_size=dynamic,
    #                     element_shape=tf.TensorShape(shape_util.shape_list(self.predict_net.get_initial_state())),
    #                     clear_after_read=False,
    #                 ),
    #             )

    #         B = initialize_beam()
    #         B = BeamHypothesis(
    #             score=B.score.write(0, 0.0),
    #             indices=B.indices.write(0, self.text_featurizer.blank),
    #             prediction=B.prediction.write(0, tf.ones([total], dtype=tf.int32) * self.text_featurizer.blank),
    #             states=B.states.write(0, self.predict_net.get_initial_state(4)),
    #         )

    #         def condition(time, total, B):
    #             return tf.less(time, total)

    #         def body(time, total, B):
    #             A = initialize_beam(dynamic=True)
    #             A = BeamHypothesis(
    #                 score=A.score.unstack(B.score.stack()),
    #                 indices=A.indices.unstack(B.indices.stack()),
    #                 prediction=A.prediction.unstack(math_util.pad_tfarray(B.prediction, blank=self.text_featurizer.blank).stack()),
    #                 states=A.states.unstack(B.states.stack()),
    #             )
    #             A_i = tf.constant(0, tf.int32)
    #             B = initialize_beam()

    #             encoded_t = tf.gather_nd(encoded, tf.expand_dims(time, axis=-1))

    #             def beam_condition(beam, beam_width, A, A_i, B):
    #                 return tf.less(beam, beam_width)

    #             def beam_body(beam, beam_width, A, A_i, B):
    #                 # get y_hat
    #                 y_hat_score, y_hat_score_index = tf.math.top_k(A.score.stack(), k=1, sorted=True)
    #                 y_hat_score = y_hat_score[0]
    #                 y_hat_index = tf.gather_nd(A.indices.stack(), y_hat_score_index)
    #                 y_hat_prediction = tf.gather_nd(
    #                     math_util.pad_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
    #                     y_hat_score_index,
    #                 )
    #                 y_hat_states = tf.gather_nd(A.states.stack(), y_hat_score_index)

    #                 # remove y_hat from A
    #                 remain_indices = tf.range(0, tf.shape(A.score.stack())[0], dtype=tf.int32)
    #                 remain_indices = tf.gather_nd(remain_indices, tf.where(tf.not_equal(remain_indices, y_hat_score_index[0])))
    #                 remain_indices = tf.expand_dims(remain_indices, axis=-1)
    #                 A = BeamHypothesis(
    #                     score=A.score.unstack(tf.gather_nd(A.score.stack(), remain_indices)),
    #                     indices=A.indices.unstack(tf.gather_nd(A.indices.stack(), remain_indices)),
    #                     prediction=A.prediction.unstack(
    #                         tf.gather_nd(
    #                             math_util.pad_tfarray(A.prediction, blank=self.text_featurizer.blank).stack(),
    #                             remain_indices,
    #                         )
    #                     ),
    #                     states=A.states.unstack(tf.gather_nd(A.states.stack(), remain_indices)),
    #                 )
    #                 A_i = tf.where(tf.equal(A_i, 0), A_i, A_i - 1)

    #                 ytu, new_states = self.decoder_inference(encoded=encoded_t, predicted=y_hat_index, states=y_hat_states, tflite=tflite)

    #                 def predict_condition(pred, A, A_i, B):
    #                     return tf.less(pred, self.text_featurizer.num_classes)

    #                 def predict_body(pred, A, A_i, B):
    #                     new_score = y_hat_score + tf.gather_nd(ytu, tf.expand_dims(pred, axis=-1))

    #                     def true_fn():
    #                         return (
    #                             B.score.write(beam, new_score),
    #                             B.indices.write(beam, y_hat_index),
    #                             B.prediction.write(beam, y_hat_prediction),
    #                             B.states.write(beam, y_hat_states),
    #                             A.score,
    #                             A.indices,
    #                             A.prediction,
    #                             A.states,
    #                             A_i,
    #                         )

    #                     def false_fn():
    #                         scatter_index = math_util.count_non_blank(y_hat_prediction, blank=self.text_featurizer.blank)
    #                         updated_prediction = tf.tensor_scatter_nd_update(
    #                             y_hat_prediction,
    #                             indices=tf.reshape(scatter_index, [1, 1]),
    #                             updates=tf.expand_dims(pred, axis=-1),
    #                         )
    #                         return (
    #                             B.score,
    #                             B.indices,
    #                             B.prediction,
    #                             B.states,
    #                             A.score.write(A_i, new_score),
    #                             A.indices.write(A_i, pred),
    #                             A.prediction.write(A_i, updated_prediction),
    #                             A.states.write(A_i, new_states),
    #                             A_i + 1,
    #                         )

    #                     b_score, b_indices, b_prediction, b_states, a_score, a_indices, a_prediction, a_states, A_i = tf.cond(
    #                         tf.equal(pred, self.text_featurizer.blank), true_fn=true_fn, false_fn=false_fn
    #                     )

    #                     B = BeamHypothesis(score=b_score, indices=b_indices, prediction=b_prediction, states=b_states)
    #                     A = BeamHypothesis(score=a_score, indices=a_indices, prediction=a_prediction, states=a_states)

    #                     return pred + 1, A, A_i, B

    #                 _, A, A_i, B = tf.while_loop(
    #                     predict_condition,
    #                     predict_body,
    #                     loop_vars=[0, A, A_i, B],
    #                     parallel_iterations=parallel_iterations,
    #                     swap_memory=swap_memory,
    #                 )

    #                 return beam + 1, beam_width, A, A_i, B

    #             _, _, A, A_i, B = tf.while_loop(
    #                 beam_condition,
    #                 beam_body,
    #                 loop_vars=[0, beam_width, A, A_i, B],
    #                 parallel_iterations=parallel_iterations,
    #                 swap_memory=swap_memory,
    #             )

    #             return time + 1, total, B

    #         _, _, B = tf.while_loop(
    #             condition,
    #             body,
    #             loop_vars=[0, total, B],
    #             parallel_iterations=parallel_iterations,
    #             swap_memory=swap_memory,
    #         )

    #         scores = B.score.stack()
    #         prediction = math_util.pad_tfarray(B.prediction, blank=self.text_featurizer.blank).stack()
    #         if self.text_featurizer.decoder_config.norm_score:
    #             prediction_lengths = math_util.count_non_blank(prediction, blank=self.text_featurizer.blank, axis=1)
    #             scores /= tf.cast(prediction_lengths, dtype=scores.dtype)

    #         y_hat_score, y_hat_score_index = tf.math.top_k(scores, k=1)
    #         y_hat_score = y_hat_score[0]
    #         y_hat_index = tf.gather_nd(B.indices.stack(), y_hat_score_index)
    #         y_hat_prediction = tf.gather_nd(prediction, y_hat_score_index)
    #         y_hat_states = tf.gather_nd(B.states.stack(), y_hat_score_index)

    #         return Hypothesis(index=y_hat_index, prediction=y_hat_prediction, states=y_hat_states)
