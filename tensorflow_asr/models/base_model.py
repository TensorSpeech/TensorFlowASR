# pylint: disable=attribute-defined-outside-init
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

# import importlib
import logging
import typing

from keras.src import tree
from keras.src.backend.tensorflow.trainer import TensorFlowTrainer, reduce_per_replica
from keras.src.losses import loss as loss_module

from tensorflow_asr import keras, schemas, tf
from tensorflow_asr.models.layers.feature_extraction import FeatureExtraction
from tensorflow_asr.optimizers.accumulation import GradientAccumulator
from tensorflow_asr.tokenizers import Tokenizer
from tensorflow_asr.utils import file_util, keras_util, math_util, shape_util

logger = logging.getLogger(__name__)


class BaseModel(keras.Model, TensorFlowTrainer):
    optimizer: typing.Union[keras.optimizers.Optimizer, keras.optimizers.LossScaleOptimizer]

    def __init__(self, speech_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extraction = FeatureExtraction(**speech_config)

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @property
    def lm(self):
        """External language model used for shallow fusion during beam search, `None` if unused"""
        return getattr(self, "_lm", None)

    def _setattr_hook(self, name, value):
        # Keep the shallow fusion LM out of keras attribute tracking. It is a frozen, externally
        # trained model: letting keras adopt it as a sub-layer would pull its weights into this
        # model's checkpoint, and assigning it after `make()` would raise outright because a built
        # layer refuses new state. `Layer.__setattr__` calls this hook before it tracks anything,
        # so stashing the LM here is the only point at which it can be hidden.
        if name == "lm":
            object.__setattr__(self, "_lm", value)
            name, value = "_lm_attached", value is not None
        return super()._setattr_hook(name, value)

    def make_lm(self, custom_objects=None):
        """
        Build the shallow fusion language model from `decoder_config.lm_config`, if one is set.

        The config is a standard keras serialization blob (`class_name` + `config`), so the class
        only has to be registered with `@keras.utils.register_keras_serializable`. No language
        model ships with TensorFlowASR -- see
        `tensorflow_asr.models.decoders.language_model.LanguageModel` for the interface to
        implement.
        """
        lm_config = getattr(getattr(self, "tokenizer", None), "decoder_config", None)
        lm_config = getattr(lm_config, "lm_config", None)
        if not lm_config:
            self.lm = None
            return None
        self.lm = keras_util.model_from_config(lm_config, custom_objects=custom_objects)
        logger.info(f"Loaded shallow fusion language model: {self.lm}")
        return self.lm

    def get_beam_decoding_kwargs(self) -> dict:
        """
        Beam search settings taken from the tokenizer's decoder config.

        Returns an empty dict when `beam_width` is not a positive number, which is the default in
        every shipped config. Callers treat that as "no beam search", matching what
        `make_tflite_function` already does with `beam_width=0`.
        """
        decoder_config = getattr(getattr(self, "tokenizer", None), "decoder_config", None)
        beam_width = int(getattr(decoder_config, "beam_width", 0) or 0)
        if beam_width <= 0:
            return {}
        kwargs = {"beam_width": beam_width, "score_norm": bool(getattr(decoder_config, "norm_score", True))}
        if self.lm is not None:
            kwargs.update(lm=self.lm, lm_alpha=float(getattr(decoder_config, "lm_alpha", 0.0)))
        return kwargs

    def summary(self, line_length=120, expand_nested=True, show_trainable=True, **kwargs):
        super().summary(line_length=line_length, expand_nested=expand_nested, show_trainable=show_trainable, **kwargs)

    def save(self, filepath, overwrite=True, zipped=None, **kwargs):
        with file_util.save_file(filepath) as path:
            super().save(filepath=path, overwrite=overwrite, zipped=zipped, **kwargs)

    def save_weights(self, filepath, overwrite=True):
        with file_util.save_file(filepath) as path:
            super().save_weights(filepath=path, overwrite=overwrite)

    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, skip_mismatch=skip_mismatch, **kwargs)

    def add_custom_metric(self, metric: keras.metrics.Metric):
        if not hasattr(self, "_tfasr_metrics"):
            self._tfasr_metrics = {}
        self._tfasr_metrics[metric.name] = metric

    def make(self, input_shape=[None], prediction_shape=[None], batch_size=None, **kwargs) -> schemas.TrainOutput:
        """
        Custom function for building model (uses self.build so cannot overwrite that function)

        Parameters
        ----------
        input_shape : list, optional
            The shape of signal, by default [None]
        prediction_shape : list, optional
            The shape of prediction, by default [None]
        batch_size : int, optional
            Batch size, by default None
        """
        assert batch_size is not None and batch_size > 0
        signals = keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        signals_length = keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self._per_replica_batch_size = int(batch_size / self.distribute_strategy.num_replicas_in_sync)
        self._batch_size = batch_size
        outputs: schemas.TrainOutput = self(
            schemas.TrainInput(
                inputs=signals,
                inputs_length=signals_length,
                predictions=predictions,
                predictions_length=predictions_length,
            ),
            training=False,
        )
        return tf.nest.map_structure(
            lambda x: shape_util.shape_list_per_replica(x, per_replica_batch_size=self._per_replica_batch_size),
            outputs,
        )  # compute output shape

    def compile(
        self,
        loss,
        optimizer=None,
        run_eagerly=False,
        ga_steps=None,
        gwn_config=None,
        gradn_config=None,
        **kwargs,
    ):
        optimizer = keras.optimizers.get(optimizer)
        if isinstance(ga_steps, int) and ga_steps > 1:
            self.use_ga = True
            self.ga = GradientAccumulator(ga_steps=ga_steps, optimizer=optimizer)
            self.ga.build(self.trainable_weights)
            kwargs["steps_per_execution"] = 1
            logger.info(f"Using gradient accumulation with accumulate steps = {ga_steps}")
        else:
            self.use_ga = False
        self.gwn_config = gwn_config
        self.gradn_config = gradn_config
        self.distribute_reduction_method = "auto"
        self.tfasr_loss = loss
        super().compile(optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs: schemas.TrainInput, training=False):
        raise NotImplementedError()

    # -------------------------------- STEP FUNCTIONS -------------------------------------
    def apply_gwn(self) -> list:
        return []

    def remove_gwn(self, original_weights):
        pass

    def tfasr_compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        loss = self.tfasr_loss(y, y_pred)
        self.add_loss(loss)
        return super()._compute_loss(x, y, y_pred, sample_weight, training)

    def _train_step(self, data: schemas.TrainData):
        x, y = data
        sample_weight = None

        with tf.GradientTape() as tape:
            tape.watch(x.inputs)
            original_weights = self.apply_gwn()
            y_pred: schemas.TrainOutput = self(x, training=True)
            tape.watch(y_pred.logits)
            self.remove_gwn(original_weights)
            loss = self.tfasr_compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )
            # loss is in shape [B]
            # reduce_mean on all replicas = (sum_loss1 / B + ... + sum_lossN / B) / N = (sum_loss1 + ... + sum_lossN) / (B * N)
            # (B * N) also total count of samples across all replicas of current batch
            # (sum_loss1 + ... + sum_lossN) is the total loss summed over all replicas of current batch, so the total number of loss = (B * N)
            # B = mini_batch_size * ga_steps
            # reduce_first = sum_loss1 / B
            # => reduce_mean has the same effect as reduce_first
            # the loss already divided by num_replicas for gradients reduce_sum when using _compute_loss, so unscale it
            self._loss_tracker.update_state(
                loss_module.unscale_loss_for_distribution(loss),
                sample_weight=tf.shape(tree.flatten(x)[0])[0],  # this is the count, which = B
            )

            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        gradients = tape.gradient(loss, self.trainable_weights)
        return gradients

    def _apply_gradients(self, gradients):
        if self.gradn_config is not None:
            gradients = tf.cond(
                tf.greater_equal(self.optimizer.iterations, self.gradn_config["step"]),
                lambda: math_util.add_gauss_noise(gradients, stddev=self.gradn_config["stddev"]),
                lambda: gradients,
            )
        self.optimizer.apply(gradients, self.trainable_weights)

    def train_step(self, data):
        gradients = self._train_step(data)
        self._apply_gradients(gradients)
        metrics = self.get_metrics_result()
        return metrics

    def train_step_ga(self, data, do_apply=None):  # avoid merge_call error as "Such behaviors are not yet supported"
        gradients = self._train_step(data)
        if do_apply is None:
            self.ga.accumulate(gradients, self.trainable_weights)
        else:
            gradients = self.ga.gradients(gradients, self.trainable_weights)
            self._apply_gradients(gradients)
            self.ga.reset()
        metrics = self.get_metrics_result()
        return metrics

    def _test_step(self, data: schemas.TrainData):
        x, y = data
        sample_weight = None
        y_pred = self(x, training=False)
        loss = self.tfasr_compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=False,
        )
        self._loss_tracker.update_state(
            loss_module.unscale_loss_for_distribution(loss),
            sample_weight=tf.shape(tree.flatten(x)[0])[0],
        )

    def test_step(self, data: schemas.TrainData):
        self._test_step(data)
        metrics = self.get_metrics_result()
        return metrics

    def predict_step(self, data: schemas.TrainData):
        x, y_true = data
        batch_size, *_ = shape_util.shape_list(x.inputs)
        inputs = schemas.PredictInput(
            inputs=x.inputs,
            inputs_length=x.inputs_length,
            previous_tokens=self.get_initial_tokens(batch_size=batch_size),
            previous_encoder_states=self.get_initial_encoder_states(batch_size=batch_size),
            previous_decoder_states=self.get_initial_decoder_states(batch_size=batch_size),
        )
        _tokens = self.recognize(inputs=inputs).tokens
        # `beam_width: 0` (the default in every shipped config) means no beam search, so the beam
        # column mirrors the greedy one instead of silently doubling the cost of every evaluation.
        _beam_kwargs = self.get_beam_decoding_kwargs()
        _beam_tokens = self.recognize_beam(inputs=inputs, **_beam_kwargs).tokens if _beam_kwargs else _tokens
        return {
            "tokens": _tokens,
            "beam_tokens": _beam_tokens,
            "labels": y_true.labels,
        }

    # ------------------------------------ FIT ----------------------------------- #

    def _make_function(self, step_function):
        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            outputs = self.distribute_strategy.run(step_function, args=(data,))
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        if not self.run_eagerly:
            one_step_on_data = tf.function(
                one_step_on_data,
                reduce_retracing=True,
                jit_compile=self.jit_compile,
            )

        def function(iterator):
            for step, data in zip(range(self.steps_per_execution), iterator):
                outputs = one_step_on_data(data)
            return outputs

        return function

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        if not self.use_ga:
            self.train_function = self._make_function(self.train_step)
            return self.train_function

        @tf.autograph.experimental.do_not_convert
        def one_ga_step_on_data(data, do_apply=None):
            outputs = self.distribute_strategy.run(self.train_step_ga, args=(data, do_apply))
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        if not self.run_eagerly:
            one_ga_step_on_data = tf.function(
                one_ga_step_on_data,
                reduce_retracing=True,
                jit_compile=self.jit_compile,
            )

        def function(iterator):
            for step, data in zip(range(self.ga.total_steps), iterator):
                if step >= self.ga.total_steps - 1:
                    outputs = one_ga_step_on_data(data, True)
                else:
                    outputs = one_ga_step_on_data(data)
            return outputs

        self.train_function = function
        return self.train_function

    # -------------------------------- INFERENCE FUNCTIONS -------------------------------------

    def get_signal_chunk_size_and_step(self, nchunks: int = 1):
        """
        How many audio samples to feed per streaming call, and how far to advance afterwards.

        Chains the two reductions between raw audio and encoder output::

            signal --FeatureExtraction--> feature frames --encoder subsampling--> encoder frames

        so it walks them backwards: `nchunks` attention chunks is
        `nchunks * chunk_size` encoder frames, which is that times `time_reduction_factor`
        feature frames, which `FeatureExtraction.get_signal_chunk_size_and_step` turns into a
        sample count.

        Getting this wrong is not a rounding error. A chunk that covers a fraction of an attention
        chunk makes the streamed decode group frames into windows a single pass never uses, and the
        two stop agreeing -- `tests/test_inference.py` asserts that from both sides. Calling this
        rather than picking a number by hand is what keeps them aligned.

        Parameters
        ----------
        nchunks : int
            Attention chunks per call. Larger means fewer, bigger calls and higher latency; the
            transcript is unchanged, since any whole number of chunks stays aligned.

        Returns
        -------
        (signal_chunk_size, signal_chunk_step)
            Samples to pass as `inputs`, and how far to advance the read position. They differ
            because consecutive frames overlap: a window is `frame_length` long but only
            `frame_step` new samples arrive each frame.
        """
        chunk_size = getattr(getattr(self, "encoder", None), "chunk_size", None) or 1
        nframes = nchunks * chunk_size * self.time_reduction_factor
        return self.feature_extraction.get_signal_chunk_size_and_step(nframes)

    def get_initial_tokens(self, batch_size=1):
        return tf.ones([batch_size, 1], dtype=tf.int32) * self.tokenizer.blank

    def get_initial_encoder_states(self, batch_size=1):
        return []

    def get_initial_decoder_states(self, batch_size=1):
        return []

    def get_initial_beam_state(self, batch_size=1, beam_width=1):
        """
        Starting state for a streaming beam: per-hypothesis score, last token and decoder state.

        Only the first hypothesis is alive, matching how `recognize_beam` seeds itself -- with W
        identical live hypotheses the first expansion would pick the same best token W times.
        Returns `(scores [B, W], last_tokens [B, W], states [B * W, ...])`.
        """
        scores = tf.concat(
            [tf.zeros([batch_size, 1], dtype=tf.float32), tf.fill([batch_size, beam_width - 1], tf.constant(-1e9, tf.float32))],
            axis=1,
        )
        last_tokens = tf.tile(self.get_initial_tokens(batch_size), [1, beam_width])
        states = tf.nest.map_structure(
            lambda state: tf.reshape(
                tf.tile(tf.expand_dims(state, axis=1), [1, beam_width, *([1] * (len(state.shape) - 1))]),
                [batch_size * beam_width, *state.shape[1:]],
            ),
            self.get_initial_decoder_states(batch_size),
        )
        return scores, last_tokens, states

    def recognize(self, inputs: schemas.PredictInput, **kwargs) -> schemas.PredictOutput:
        """Greedy decoding function that used in self.predict_step"""
        raise NotImplementedError()

    def recognize_beam(self, inputs: schemas.PredictInput, beam_width: int = 10, **kwargs) -> schemas.PredictOutput:
        """Beam search decoding function that used in self.predict_step"""
        raise NotImplementedError()

    # ---------------------------------- TFLITE ---------------------------------- #

    def make_tflite_function(self, batch_size: int = 1, beam_width: int = 0):

        def tflite_func(inputs: schemas.PredictInput):
            if beam_width > 0:
                outputs = self.recognize_beam(inputs, beam_width=beam_width)
            else:
                outputs = self.recognize(inputs)
            return schemas.PredictOutputWithTranscript(
                transcript=self.tokenizer.detokenize(outputs.tokens),
                tokens=outputs.tokens,
                next_tokens=outputs.next_tokens,
                next_encoder_states=outputs.next_encoder_states,
                next_decoder_states=outputs.next_decoder_states,
                next_beam_scores=outputs.next_beam_scores,
                next_beam_last_tokens=outputs.next_beam_last_tokens,
                next_beam_states=outputs.next_beam_states,
            )

        beam_signature = {}
        if beam_width > 0:
            # Only the beam export carries these, so a greedy signature is byte-for-byte what it
            # was before streaming beam search existed.
            beam_scores, beam_last_tokens, beam_states = self.get_initial_beam_state(batch_size, beam_width)
            beam_signature = dict(
                previous_beam_scores=tf.TensorSpec.from_tensor(beam_scores),
                previous_beam_last_tokens=tf.TensorSpec.from_tensor(beam_last_tokens),
                previous_beam_states=tf.nest.map_structure(tf.TensorSpec.from_tensor, beam_states),
            )

        input_signature = schemas.PredictInput(
            inputs=tf.TensorSpec([batch_size, None], dtype=tf.float32),
            inputs_length=tf.TensorSpec([batch_size], dtype=tf.int32),
            previous_tokens=tf.TensorSpec.from_tensor(self.get_initial_tokens(batch_size)),
            previous_encoder_states=tf.nest.map_structure(tf.TensorSpec.from_tensor, self.get_initial_encoder_states(batch_size)),
            previous_decoder_states=tf.nest.map_structure(tf.TensorSpec.from_tensor, self.get_initial_decoder_states(batch_size)),
            **beam_signature,
        )

        return tf.function(
            tflite_func,
            input_signature=[input_signature],
            jit_compile=True,
            reduce_retracing=True,
            autograph=True,
        )
