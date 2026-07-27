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
"""https://arxiv.org/pdf/1811.06621.pdf"""

import collections
import typing

from keras.src import backend

from tensorflow_asr import keras, schemas, tf
from tensorflow_asr.losses.rnnt_loss import RnntLoss
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.layers.embedding import Embedding, OneHotBlank
from tensorflow_asr.models.layers.general import Activation
from tensorflow_asr.utils import env_util, layer_util, shape_util

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))

JOINT_MODES = ["add", "mul"]


def _tile_to_beam(states: tf.Tensor, beam: int):
    """Replicate per-utterance states across the beam: [B, ...] -> [B * W, ...]"""
    batch_size, *trailing = shape_util.shape_list(states)
    tiled = tf.tile(tf.expand_dims(states, axis=1), [1, beam, *([1] * len(trailing))])
    return tf.reshape(tiled, [batch_size * beam, *trailing])


def _select_beam_states(previous: tf.Tensor, updated: tf.Tensor, parents: tf.Tensor, keep_previous: tf.Tensor, batch_size, beam: int):
    """
    Re-order per-hypothesis states onto the newly selected parents.

    `previous` and `updated` are [B * W, ...] with any trailing rank, `parents` and `keep_previous`
    are [B, W]. A blank expansion does not advance a network, so it keeps its parent's states.
    """
    _, *trailing = shape_util.shape_list(previous)
    grouped = [batch_size, beam, *trailing]
    selected = tf.where(
        tf.reshape(keep_previous, [batch_size, beam, *([1] * len(trailing))]),
        tf.gather(tf.reshape(previous, grouped), parents, batch_dims=1),
        tf.gather(tf.reshape(updated, grouped), parents, batch_dims=1),
    )
    return tf.reshape(selected, [batch_size * beam, *trailing])


def _pick_beam_states(states: tf.Tensor, indices: tf.Tensor, batch_size, beam: int):
    """Pick one hypothesis per utterance out of [B * W, ...] using [B, 2] indices -> [B, ...]"""
    _, *trailing = shape_util.shape_list(states)
    return tf.gather_nd(tf.reshape(states, [batch_size, beam, *trailing]), indices)


def _internal_lm_log_probs(logits: tf.Tensor, blank: int, vocab_size):
    """
    Turn joint-network logits into internal language model log-probabilities.

    The caller produces `logits` by running the joint with the encoder output zeroed, so what is
    left is the label-history path alone -- `z_ILM = J(g_u) = W_j phi(W_p h_pred + b_p) + b_j`, eq.
    (25) of the ILME paper (https://arxiv.org/abs/2011.01991). Two steps remain:

    1. Drop the blank logit. A language model has no blank symbol, and leaving it in the
       normalisation would make every label probability depend on a token the LM cannot emit.
       Masking it to a large negative number before the softmax is equivalent to normalising over
       the labels alone, and keeps the shape static for XLA / TFLite.
    2. Give the blank column the value 0.0 on the way out. The fusion below subtracts this tensor,
       and blank must not be touched by the subtraction -- so 0.0 is the identity, not a
       probability.

    `logits` is [..., V], the result has the same shape.
    """
    is_blank = tf.equal(tf.range(vocab_size, dtype=tf.int32), blank)  # [V]
    # `dtype.min` rather than a literal: this runs at the model's compute dtype, and a hardcoded
    # -1e9 silently overflows to -inf under mixed_float16.
    masked = tf.where(is_blank, tf.constant(logits.dtype.min, dtype=logits.dtype), logits)
    return tf.where(is_blank, tf.zeros_like(logits), tf.nn.log_softmax(masked))


def _fuse_lm(
    log_probs: tf.Tensor,
    lm_log_probs: tf.Tensor,
    ilm_log_probs: tf.Tensor,
    lm_alpha: float,
    lm_beta: float,
    blank: int,
    vocab_size,
):
    """
    Fuse an external LM into the transducer log-probabilities and optionally subtract an estimate
    of the transducer's own internal LM:

        ln p_tot[k] = ln p[k] + a * (ln (1 - p[blank]) + ln p_lm[k]) - b * ln p_ilm[k]  for k != blank
        ln p_tot[blank] = (1 + a) * ln p[blank]

    The `a` term is eq. (3) of the ALSD++ paper (https://arxiv.org/abs/2506.00185). Scaling blank by
    (1 + a) rather than leaving it alone is the point of that formulation: boosting only the label
    scores would make blank comparatively cheaper at every frame and drive the deletion rate up.

    The `b` term is the internal LM correction of eq. (27) of the ILME paper
    (https://arxiv.org/abs/2011.01991). A transducer trained on paired speech and text learns a
    language model of the training transcripts whether or not anyone asked for one, and that
    implicit model fights the external LM on any domain it was not trained on. Subtracting it lets
    the external LM speak for itself. Where `ln p_ilm` comes from is the caller's choice:

    - ILME reads it off the transducer itself, by running the joint with the encoder output zeroed.
      Exact, and free of extra parameters, but it costs a second joint call per step.
    - LODR (https://arxiv.org/abs/2203.16776) replaces it with a cheap low-order n-gram LM trained
      on the same transcripts. Only an approximation of the internal LM, but a bigram costs a table
      lookup instead of a joint call, and the paper reports it matching ILME in practice.

    Both weights default to a no-op: `a = 0` leaves the transducer log-probabilities alone, and
    `b = 0` subtracts nothing.

    `log_probs`, `lm_log_probs` and `ilm_log_probs` are [B, W, V]. Either LM tensor may be `None`,
    meaning that half of the formula is dropped rather than multiplied by zero -- so a setup that
    only shallow fuses builds exactly the graph it built before internal LM subtraction existed.
    The external LM's blank column is never read; the internal LM's blank column is 0.0 by
    construction, see `_internal_lm_log_probs`.
    """
    blank_log_prob = log_probs[..., blank : blank + 1]  # [B, W, 1]
    fused_labels = log_probs
    fused_blank = blank_log_prob
    if lm_log_probs is not None:
        # ln(1 - p[blank]) as ln(-expm1(x)), the stable form of log1mexp for x < 0. The clamp keeps
        # the argument strictly negative so a saturated p[blank] = 1 cannot produce ln(0) = -inf.
        log_not_blank = tf.math.log(-tf.math.expm1(tf.minimum(blank_log_prob, -1e-7)))
        fused_labels = fused_labels + lm_alpha * (log_not_blank + lm_log_probs)
        fused_blank = (1.0 + lm_alpha) * blank_log_prob
    if ilm_log_probs is not None:
        fused_labels = fused_labels - lm_beta * ilm_log_probs
    is_blank = tf.equal(tf.range(vocab_size, dtype=tf.int32), blank)  # [V]
    return tf.where(is_blank, tf.broadcast_to(fused_blank, tf.shape(log_probs)), fused_labels)


@keras.utils.register_keras_serializable(package=__name__)
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
        activity_regularizer=None,
        recurrent_regularizer=None,
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
        self.rnns: typing.List[typing.Union[keras.layers.GRU, keras.layers.LSTM, keras.layers.SimpleRNN]] = []
        self.lns = []
        self.projections = []
        for i in range(num_rnns):
            rnn = layer_util.get_rnn(rnn_type)(
                units=rnn_units,
                return_sequences=True,
                name=f"{rnn_type}_{i}",
                return_state=True,
                implementation=rnn_implementation,
                unroll=rnn_unroll,
                zero_output_for_mask=True,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                use_cudnn=env_util.TF_CUDNN,
                dtype=self.dtype,
            )
            ln = (
                keras.layers.LayerNormalization(
                    name=f"ln_{i}", gamma_regularizer=kernel_regularizer, beta_regularizer=kernel_regularizer, dtype=self.dtype
                )
                if layer_norm
                else None
            )
            projection = (
                keras.layers.Dense(
                    projection_units,
                    name=f"projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
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
            states.append(tf.stack(rnn.get_initial_state(batch_size=batch_size), axis=0))
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


@keras.utils.register_keras_serializable(package=__name__)
class TransducerJointMerge(Layer):
    def __init__(self, joint_mode: str = "add", name="transducer_joint_merge", **kwargs):
        super().__init__(name=name, **kwargs)
        if joint_mode not in JOINT_MODES:
            raise ValueError(f"joint_mode must in {JOINT_MODES}")
        self.joint_mode = joint_mode

    def compute_mask(self, inputs, mask=None):
        enc_out, pred_out = inputs
        enc_mask = mask[0] if mask else backend.get_keras_mask(enc_out)  # BT
        pred_mask = mask[1] if mask else backend.get_keras_mask(pred_out)  # BU
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


@keras.utils.register_keras_serializable(package=__name__)
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
        activity_regularizer=None,
        name="tranducer_joint",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.prejoint_encoder_linear = prejoint_encoder_linear
        self.prejoint_prediction_linear = prejoint_prediction_linear
        self.postjoint_linear = postjoint_linear

        if self.prejoint_encoder_linear:
            self.ffn_enc = keras.layers.Dense(
                joint_dim,
                name="enc",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                dtype=self.dtype,
            )
        if self.prejoint_prediction_linear:
            self.ffn_pred = keras.layers.Dense(
                joint_dim,
                name="pred",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                dtype=self.dtype,
            )

        self.joint = TransducerJointMerge(joint_mode=joint_mode, name="merge", dtype=self.dtype)

        activation = activation.lower()
        self.activation = Activation(activation, name=activation, dtype=self.dtype)

        if self.postjoint_linear:
            self.ffn = keras.layers.Dense(
                joint_dim,
                name="ffn",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                dtype=self.dtype,
            )

        self.ffn_out = keras.layers.Dense(
            vocab_size,
            name="vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
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
        encoder: Layer,
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
        activity_regularizer=None,
        recurrent_regularizer=None,
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
            activity_regularizer=activity_regularizer,
            recurrent_regularizer=recurrent_regularizer,
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
            activity_regularizer=activity_regularizer,
            trainable=joint_trainable,
            name="joint",
            dtype=self.dtype,
        )
        self.time_reduction_factor = 1

    def compile(self, optimizer, output_shapes=None, **kwargs):
        loss = RnntLoss(blank=self.blank, output_shapes=output_shapes, name="rnnt_loss")
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

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
        features, features_length = self.feature_extraction((inputs.inputs, inputs.inputs_length), training=training)
        enc, logits_length, *_ = self.encoder((features, features_length), training=training)
        pred, *_ = self.predict_net((inputs.predictions, inputs.predictions_length), training=training)
        logits = self.joint_net((enc, pred), training=training)
        return schemas.TrainOutput(
            logits=logits,
            logits_length=logits_length,
        )

    def call_next(
        self,
        current_frames: tf.Tensor,
        previous_tokens: tf.Tensor,
        previous_decoder_states: tf.Tensor,
        return_internal_lm: bool = False,
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
        return_internal_lm : bool
            Also return the internal LM log-probabilities, for the ILME correction of
            https://arxiv.org/abs/2011.01991. This is a python flag read at trace time, so it costs
            nothing when off.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor], shapes ([B, 1, 1, V], [B, num_rnns, nstates, state_size])
            Output of joint network of the current frame, new states of prediction network.
            With `return_internal_lm`, the internal LM log-probabilities [B, 1, 1, V] are inserted
            in the middle, making it a 3-tuple.
        """
        with tf.name_scope(f"{self.name}_call_next"):
            y, new_states = self.predict_net.call_next(previous_tokens, previous_decoder_states)
            ytu = self.joint_net([current_frames, y], training=False)
            ytu = tf.nn.log_softmax(ytu)
            if not return_internal_lm:
                return ytu, new_states
            # Zeroing the encoder output is what isolates the internal LM: the joint is
            # `ffn_out(act(merge(ffn_enc(enc), ffn_pred(pred))))`, so with `enc = 0` and the default
            # additive merge only `ffn_enc`'s bias survives from the acoustic side, leaving exactly
            # the `W_j phi(W_p h_pred + b_p) + b_j` of eq. (25) in the ILME paper. Reusing `y` is the
            # reason this lives here rather than in a method of its own -- the prediction network is
            # the expensive part of a decoding step and it must not be run twice.
            ilm = self.joint_net([tf.zeros_like(current_frames), y], training=False)
            vocab_size = shape_util.shape_list(ilm)[-1]
            return ytu, _internal_lm_log_probs(ilm, self.blank, vocab_size), new_states

    def get_initial_encoder_states(self, batch_size=1):
        return []

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

    def recognize_beam(
        self,
        inputs: schemas.PredictInput,
        beam_width: int = 10,
        max_tokens_per_frame: int = 3,
        score_norm: bool = True,
        lm=None,
        lm_alpha: float = 0.0,
        lm_type: str = "shallow",
        internal_lm=None,
        lm_beta: float = 0.0,
        **kwargs,
    ):
        """
        ALSD++ beam search decoding.

        Ref:
            [1] "Pushing the Limits of Beam Search Decoding for Transducer-based ASR models",
                L. Grigoryan et al., Interspeech 2025, https://arxiv.org/abs/2506.00185
            [2] "Alignment-Length Synchronous Decoding for RNN Transducer" (the original ALSD),
                G. Saon et al., ICASSP 2020
            [3] "Sequence Transduction with Recurrent Neural Networks" (the original transducer
                beam search), A. Graves, 2012, https://arxiv.org/abs/1211.3711
            [4] "Internal Language Model Estimation for Domain-Adaptive End-to-End Speech
                Recognition" (ILME), Z. Meng et al., SLT 2021, https://arxiv.org/abs/2011.01991
            [5] "Low-order Density Ratio: ... " (LODR), Z. Yao et al., 2022,
                https://arxiv.org/abs/2203.16776

        Every iteration advances *each* hypothesis by exactly one step in the transducer lattice,
        either along the time axis (blank) or along the label axis (non-blank). All hypotheses in a
        beam therefore always share the same alignment length t + u, which is what makes their
        accumulated log-probabilities directly comparable -- the defining property of ALSD [2].

        ALSD++ [1] replaces the fixed S = T + U_max iteration budget of [2] with a frame driven
        bound: the loop runs until every hypothesis has consumed all T encoder frames, and each
        hypothesis may emit at most `max_tokens_per_frame` (the `s` of [1]) labels per frame. This
        stops hypotheses that already reached the end of the audio from burning the remaining
        iterations on hallucinated tokens.

        Deviation from [1]: the reference implementation stores transcripts in a trie
        (`transcripts` + `transcripts_ptrs` backlinks) to avoid copying whole transcripts on each
        expansion. Here the transcripts are kept dense and re-gathered per step, because a
        `tf.gather` over the beam axis is a single vectorized op and keeps every shape static, which
        is what TFLite / XLA export needs. The hash based recombination of [1] is kept as-is.

        Parameters
        ----------
        inputs : schemas.PredictInput
        beam_width : int
            Number of hypotheses kept per utterance (the `W` below).
        max_tokens_per_frame : int
            `s` in [1], the maximum number of non-blank expansions allowed on a single frame.
        score_norm : bool
            Divide the final score by the transcript length before picking the winner, to
            counteract the bias of accumulated log-probabilities towards short transcripts.
        lm : Optional[LanguageModel]
            External language model to shallow fuse, see
            `tensorflow_asr.models.lm.language_model.LanguageModel`. `None` disables fusion
            entirely, so no LM call is made.
        lm_alpha : float
            `lambda` of eq. (3) in [1], the shallow fusion weight. `0.0` makes fusion a no-op
            mathematically, but the LM is still evaluated -- pass `lm=None` to skip the work.
        lm_type : str
            How to correct for the transducer's own internal language model, see `_fuse_lm`:

            - "shallow": no correction, external LM only. The default.
            - "ilme":    subtract the exact internal LM read off the transducer, by running the
                         joint a second time with the encoder output zeroed [4].
            - "lodr":    subtract `internal_lm`, a cheap low-order n-gram standing in for the
                         internal LM [5].
        internal_lm : Optional[LanguageModel]
            The low-order LM of "lodr", same interface as `lm`. Ignored by the other types.
        lm_beta : float
            The internal LM weight, `lambda_I` of eq. (27) in [4]. Ignored when `lm_type` is
            "shallow". Both papers tune it below the external weight: [4] lands on
            `lambda_I / lambda_T` between 0.375 and 0.77 across its RNN-T setups, [5] on roughly
            0.2. Neither paper analyses the failure mode, but it follows from `_fuse_lm`:
            `ln p_ilm` is negative, so subtracting it *raises* label scores while blank is left
            alone, and oversubtracting therefore drives over-emission.

        Returns
        -------
        schemas.PredictOutput
            Same contract as `recognize`, the values are taken from the best scoring hypothesis.
        """
        with tf.name_scope(f"{self.name}_recognize_beam"):
            if lm_type not in ("shallow", "ilme", "lodr"):
                raise ValueError(f'lm_type must be one of "shallow", "ilme", "lodr", got "{lm_type}"')
            if lm_type == "lodr" and internal_lm is None:
                raise ValueError('lm_type "lodr" needs an `internal_lm` to subtract, got None')
            # Stand-in for -inf: kept finite so that masked entries can be added to without
            # producing NaNs, and small enough that they can never win a top_k.
            neg_inf = tf.constant(-1e9, dtype=tf.float32)
            # Rolling hash constants, see "hash-based transcript representations" in [1]:
            # H_{u+1} = (H_u * P + T_{u+1}) mod M
            hash_prime = tf.constant(1_000_003, dtype=tf.int64)
            hash_modulo = tf.constant(1_000_000_007, dtype=tf.int64)

            features, features_length = self.feature_extraction((inputs.inputs, inputs.inputs_length), training=False)
            encoded, encoded_length, next_encoder_states = self.encoder.call_next(features, features_length, inputs.previous_encoder_states)

            batch_size, max_frames, _ = shape_util.shape_list(encoded)
            beam = beam_width
            batch_beam = batch_size * beam
            # Same bound as the greedy batch decoder, so both branches emit the same output width
            max_tokens = max_frames * 2 + 1
            nframes = tf.reshape(encoded_length, [batch_size, 1])  # [B, 1]
            last_frame = tf.maximum(nframes - 1, 0)  # [B, 1]

            # The beam is folded into the batch axis, so the prediction and joint networks are
            # called once for all B * W hypotheses -- the "batch operations" of [1]
            # A carried beam resumes all W hypotheses; otherwise the beam starts from the single
            # greedy state. Streaming without this collapses the beam to width one at every chunk
            # boundary, so each chunk re-explores from one hypothesis instead of W.
            # `()` is the "absent" marker (see schemas.PredictInput) -- a tensor is always present
            resumed = not isinstance(inputs.previous_beam_scores, (list, tuple)) and inputs.previous_beam_scores is not None
            if resumed:
                states = inputs.previous_beam_states  # [B * W, num_rnns, nstates, state_size]
                last_tokens = inputs.previous_beam_last_tokens  # [B, W]
                scores = inputs.previous_beam_scores  # [B, W]
            else:
                states = _tile_to_beam(inputs.previous_decoder_states, beam)
                last_tokens = tf.tile(inputs.previous_tokens, [1, beam])  # [B, W]
                # Only the first hypothesis is alive initially, otherwise the first expansion would
                # select the same best token W times over W identical hypotheses
                scores = tf.concat([tf.zeros([batch_size, 1], dtype=tf.float32), tf.fill([batch_size, beam - 1], neg_inf)], axis=1)  # [B, W]
            # Shallow fusion state, threaded through the beam exactly like the prediction network
            # state. With no LM, a scalar placeholder keeps the loop signature uniform.
            lm_states = _tile_to_beam(lm.get_initial_state(batch_size) if lm is not None else tf.zeros([batch_size, 1]), beam)
            # Same for the LODR low-order LM. "ilme" needs no state of its own: it reads the
            # internal LM off the prediction network, whose state is already carried in `states`.
            _has_lodr = lm_type == "lodr"
            ilm_states = _tile_to_beam(internal_lm.get_initial_state(batch_size) if _has_lodr else tf.zeros([batch_size, 1]), beam)
            frame_indices = tf.zeros([batch_size, beam], dtype=tf.int32)  # t of each hypothesis
            num_expansions = tf.zeros([batch_size, beam], dtype=tf.int32)  # labels emitted on the current frame
            tokens = tf.ones([batch_size, beam, max_tokens], dtype=tf.int32) * self.blank
            tokens_length = tf.zeros([batch_size, beam], dtype=tf.int32)  # u of each hypothesis
            if resumed:
                # A resumed beam starts each chunk with an empty transcript, so every hypothesis
                # would hash to the same value and the recombination step -- which drops the later
                # of any two hypotheses sharing a hash -- would collapse the carried beam back to a
                # single slot, silently undoing the carry. Seeding one distinct hash per slot keeps
                # them apart. It is the conservative choice: hypotheses whose histories really did
                # converge before the boundary stay separate rather than merging, which costs a
                # little beam diversity but can never merge two genuinely different histories.
                hashes = tf.tile(tf.reshape(tf.range(beam, dtype=tf.int64), [1, beam]), [batch_size, 1])
            else:
                hashes = tf.zeros([batch_size, beam], dtype=tf.int64)

            # The set of completed hypotheses, `F` in [2], kept outside the beam. Only the best
            # element of `F` is ever returned, so a running argmax is equivalent to materialising
            # the whole set. See the harvest step in the loop body for why `F` is needed at all.
            final_scores = tf.fill([batch_size], neg_inf)  # [B]
            final_tokens = tf.ones([batch_size, max_tokens], dtype=tf.int32) * self.blank
            final_last_tokens = tf.reshape(inputs.previous_tokens, [batch_size])  # [B]
            final_states = inputs.previous_decoder_states

            # Indices of the [B, W] grid, reused to scatter one token per hypothesis per step
            grid_batch = tf.tile(tf.reshape(tf.range(batch_size, dtype=tf.int32), [batch_size, 1]), [1, beam])
            grid_beam = tf.tile(tf.reshape(tf.range(beam, dtype=tf.int32), [1, beam]), [batch_size, 1])
            # True where i < j, used to drop the worse copy of a pair of duplicate hypotheses
            _rows = tf.range(beam, dtype=tf.int32)
            strictly_before = tf.less(tf.expand_dims(_rows, axis=1), tf.expand_dims(_rows, axis=0))  # [W, W]

            def cond(
                _frame_indices,
                _num_expansions,
                _last_tokens,
                _states,
                _scores,
                _tokens,
                _tokens_length,
                _hashes,
                _final_scores,
                _final_tokens,
                _final_last_tokens,
                _final_states,
                _lm_states,
                _ilm_states,
            ):
                # ALSD++ terminates on frames consumed, not on a fixed alignment length [1]
                return tf.logical_not(tf.math.reduce_all(tf.greater_equal(_frame_indices, nframes)))

            def body(
                _frame_indices,
                _num_expansions,
                _last_tokens,
                _states,
                _scores,
                _tokens,
                _tokens_length,
                _hashes,
                _final_scores,
                _final_tokens,
                _final_last_tokens,
                _final_states,
                _lm_states,
                _ilm_states,
            ):
                ##################### joint network, one call for the whole B * W beam
                _current_frames = tf.gather(encoded, tf.minimum(_frame_indices, last_frame), batch_dims=1)  # [B, W, E]
                _current_frames = tf.reshape(_current_frames, [batch_beam, 1, -1])  # [B * W, 1, E]
                _previous = tf.reshape(_last_tokens, [batch_beam, 1])  # [B * W, 1]
                if lm_type == "ilme":
                    _log_probs, _ilm_log_probs, _new_states = self.call_next(_current_frames, _previous, _states, return_internal_lm=True)
                    _ilm_log_probs = tf.reshape(tf.cast(_ilm_log_probs, tf.float32), [batch_size, beam, -1])  # [B, W, V]
                else:
                    _log_probs, _new_states = self.call_next(_current_frames, _previous, _states)
                    _ilm_log_probs = None
                _log_probs = tf.reshape(tf.cast(_log_probs, tf.float32), [batch_size, beam, -1])  # [B, W, V]
                _vocab_size = shape_util.shape_list(_log_probs)[-1]

                ##################### language model fusion
                # `lm`, `lm_type` and `internal_lm` are python objects known at trace time, so every
                # branch here is resolved while tracing: an absent LM costs nothing at all rather
                # than a masked-out branch inside the graph.
                if lm is None:
                    _lm_updated = _lm_states
                    _lm_log_probs = None
                else:
                    _lm_log_probs, _lm_updated = lm.call_next(_previous, _lm_states)
                    _lm_log_probs = tf.reshape(tf.cast(_lm_log_probs, tf.float32), [batch_size, beam, -1])  # [B, W, V]
                if _has_lodr:
                    _ilm_log_probs, _ilm_updated = internal_lm.call_next(_previous, _ilm_states)
                    _ilm_log_probs = tf.reshape(tf.cast(_ilm_log_probs, tf.float32), [batch_size, beam, -1])  # [B, W, V]
                    # The low-order LM is a plain LM, so it has no blank column to speak of. Zeroing
                    # it makes the subtraction a no-op at blank, matching `_internal_lm_log_probs`.
                    _ilm_log_probs = tf.where(
                        tf.equal(tf.range(_vocab_size, dtype=tf.int32), self.blank),
                        tf.zeros_like(_ilm_log_probs),
                        _ilm_log_probs,
                    )
                else:
                    _ilm_updated = _ilm_states
                if _lm_log_probs is not None or _ilm_log_probs is not None:
                    _log_probs = _fuse_lm(_log_probs, _lm_log_probs, _ilm_log_probs, lm_alpha, lm_beta, self.blank, _vocab_size)

                ##################### forced blanks
                # `_blocked` leaves the blank log-probability untouched and takes every label out
                # of contention: [0, ..., -inf, ..., 0] with the zero at the blank index.
                _blocked = tf.one_hot(self.blank, depth=_vocab_size, on_value=0.0, off_value=neg_inf, dtype=tf.float32)  # [V]
                # A hypothesis that hit the per-frame cap `s` has to move on to the next frame --
                # the ALSD++ anti-hallucination constraint [1]. It still pays the real ln p(blank)
                # for consuming the frame, otherwise emitting the full `s` labels would buy a free
                # frame transition and the search would be biased towards over-emitting.
                _capped = tf.logical_or(
                    tf.greater_equal(_num_expansions, max_tokens_per_frame),
                    tf.greater_equal(_tokens_length, max_tokens),
                )
                _log_probs = tf.where(tf.expand_dims(_capped, axis=-1), tf.add(_log_probs, _blocked), _log_probs)
                # A hypothesis that consumed every frame is complete, so its score is final: the
                # leftover iterations must leave it untouched, ie. blank at log-probability 0.
                _exhausted = tf.greater_equal(_frame_indices, nframes)
                _log_probs = tf.where(tf.expand_dims(_exhausted, axis=-1), _blocked, _log_probs)

                ##################### recombination
                # Hypotheses spelling the same transcript differ only in blank placement, so they
                # are the same hypothesis; [1] compares them in constant time through the rolling
                # hash and kills the duplicates by setting their score to -inf. Hypotheses are
                # sorted by score after top_k, so the survivor is always the best scoring one.
                _same_hash = tf.equal(tf.expand_dims(_hashes, axis=2), tf.expand_dims(_hashes, axis=1))  # [B, W, W]
                _is_duplicate = tf.math.reduce_any(tf.logical_and(_same_hash, strictly_before), axis=1)  # [B, W]
                _scores = tf.where(_is_duplicate, neg_inf, _scores)

                ##################### expansion, blank and non-blank compete in one top_k
                # Blank moves a hypothesis to the next frame, a label extends it on the same frame.
                # Both are a single lattice step, so all W survivors keep the same alignment length.
                _candidates = tf.expand_dims(_scores, axis=-1) + _log_probs  # [B, W, V]

                ##################### harvest completed hypotheses into `F`
                # A candidate is complete exactly when it takes the blank of the last frame, and it
                # has to be harvested here, from the full W * V candidate set, rather than after the
                # prune below: a complete hypothesis stops accumulating log-probabilities while the
                # partial ones keep going, so it is not guaranteed to rank inside the top W at the
                # step it completes, and pruning it there would lose it for good.
                # Blank carries the parent's transcript, last token and prediction states over
                # unchanged, so the pre-prune tensors are already the right ones to record.
                _completes = tf.greater_equal(tf.add(_frame_indices, 1), nframes)  # [B, W]
                _ranked = _candidates[:, :, self.blank]  # [B, W]
                if score_norm:
                    _ranked = tf.divide(_ranked, tf.cast(tf.maximum(_tokens_length, 1), _ranked.dtype))
                _ranked = tf.where(_completes, _ranked, neg_inf)
                _top = tf.stack([tf.range(batch_size, dtype=tf.int32), tf.argmax(_ranked, axis=1, output_type=tf.int32)], axis=-1)  # [B, 2]
                _better = tf.greater(tf.gather_nd(_ranked, _top), _final_scores)  # [B]
                _final_scores = tf.where(_better, tf.gather_nd(_ranked, _top), _final_scores)
                _final_tokens = tf.where(tf.expand_dims(_better, axis=-1), tf.gather_nd(_tokens, _top), _final_tokens)
                _final_last_tokens = tf.where(_better, tf.gather_nd(_last_tokens, _top), _final_last_tokens)
                _final_states = tf.where(
                    tf.reshape(_better, [batch_size, 1, 1, 1]),
                    _pick_beam_states(_states, _top, batch_size, beam),
                    _final_states,
                )

                ##################### prune to the W best candidates
                _scores, _candidate_indices = tf.math.top_k(tf.reshape(_candidates, [batch_size, -1]), k=beam, sorted=True)
                _parents = tf.math.floordiv(_candidate_indices, _vocab_size)  # [B, W]
                _emitted = tf.math.floormod(_candidate_indices, _vocab_size)  # [B, W]
                _is_blank = tf.equal(_emitted, self.blank)  # [B, W]

                ##################### rebuild the beam from the selected parents
                _tokens = tf.gather(_tokens, _parents, batch_dims=1)
                _tokens_length = tf.gather(_tokens_length, _parents, batch_dims=1)
                _frame_indices = tf.gather(_frame_indices, _parents, batch_dims=1)
                _num_expansions = tf.gather(_num_expansions, _parents, batch_dims=1)
                _hashes = tf.gather(_hashes, _parents, batch_dims=1)
                _last_tokens = tf.gather(_last_tokens, _parents, batch_dims=1)
                # Blank advances neither the prediction network nor the LMs, so it keeps the parent states
                _states = _select_beam_states(_states, _new_states, _parents, _is_blank, batch_size, beam)
                _lm_states = _select_beam_states(_lm_states, _lm_updated, _parents, _is_blank, batch_size, beam)
                _ilm_states = _select_beam_states(_ilm_states, _ilm_updated, _parents, _is_blank, batch_size, beam)

                ##################### append the emitted label
                _write_indices = tf.stack([grid_batch, grid_beam, tf.minimum(_tokens_length, max_tokens - 1)], axis=-1)  # [B, W, 3]
                _updates = tf.where(
                    tf.logical_or(_is_blank, tf.greater_equal(_tokens_length, max_tokens)),
                    tf.gather_nd(_tokens, _write_indices),  # rewrite the current value, ie. no-op
                    _emitted,
                )
                _tokens = tf.tensor_scatter_nd_update(_tokens, _write_indices, _updates)
                _tokens_length = tf.where(_is_blank, _tokens_length, tf.minimum(tf.add(_tokens_length, 1), max_tokens))
                _hashes = tf.where(
                    _is_blank,
                    _hashes,
                    tf.math.floormod(tf.add(tf.multiply(_hashes, hash_prime), tf.cast(_emitted, tf.int64) + 1), hash_modulo),
                )

                ##################### step updates
                _frame_indices = tf.where(_is_blank, tf.add(_frame_indices, 1), _frame_indices)
                _num_expansions = tf.where(_is_blank, tf.zeros_like(_num_expansions), tf.add(_num_expansions, 1))
                _last_tokens = tf.where(_is_blank, _last_tokens, _emitted)

                return (
                    _frame_indices,
                    _num_expansions,
                    _last_tokens,
                    _states,
                    _scores,
                    _tokens,
                    _tokens_length,
                    _hashes,
                    _final_scores,
                    _final_tokens,
                    _final_last_tokens,
                    _final_states,
                    _lm_states,
                    _ilm_states,
                )

            (
                frame_indices,
                num_expansions,
                last_tokens,
                states,
                scores,
                tokens,
                tokens_length,
                hashes,
                final_scores,
                final_tokens,
                final_last_tokens,
                final_states,
                lm_states,
                ilm_states,
            ) = tf.while_loop(
                cond,
                body,
                loop_vars=(
                    frame_indices,
                    num_expansions,
                    last_tokens,
                    states,
                    scores,
                    tokens,
                    tokens_length,
                    hashes,
                    final_scores,
                    final_tokens,
                    final_last_tokens,
                    final_states,
                    lm_states,
                    ilm_states,
                ),
                # Each hypothesis emits at most `s` labels before being forced to consume a frame,
                # so T * (s + 1) steps are enough to drain every frame -- the static bound of [1]
                maximum_iterations=max_frames * (max_tokens_per_frame + 1),
                back_prop=False,
            )

            return schemas.PredictOutput(
                tokens=final_tokens,
                next_tokens=tf.reshape(final_last_tokens, [batch_size, 1]),
                next_encoder_states=next_encoder_states,
                next_decoder_states=final_states,
                # The whole beam, so the next chunk can continue all W hypotheses. `scores` is
                # rebased on the best hypothesis: the absolute log-probability grows without bound
                # over a long stream and only the differences between hypotheses matter, so
                # subtracting the maximum keeps the numbers bounded without changing any ranking.
                next_beam_scores=scores - tf.reduce_max(scores, axis=1, keepdims=True),
                next_beam_last_tokens=last_tokens,
                next_beam_states=states,
            )

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
    #         return Hypothesis(index=y_hat_index, prediction=y_hat_prediction, states=y_hat_states)
