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

import importlib
import logging

import numpy as np

from tensorflow_asr import keras, schemas, tf
from tensorflow_asr.models.layers.feature_extraction import FeatureExtraction
from tensorflow_asr.optimizers.accumulation import GradientAccumulator
from tensorflow_asr.tokenizers import Tokenizer
from tensorflow_asr.utils import data_util, env_util, file_util, keras_util, shape_util

tf_utils = importlib.import_module(f"{env_util.KERAS_SRC}.utils.tf_utils")
io_utils = importlib.import_module(f"{env_util.KERAS_SRC}.utils.io_utils")
_minimum_control_deps = importlib.import_module(f"{env_util.KERAS_SRC}.engine.training")._minimum_control_deps

logger = logging.getLogger(__name__)


class BaseModel(keras.Model):
    def __init__(self, speech_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extraction = FeatureExtraction(**speech_config, dtype=self.dtype)

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def summary(
        self,
        line_length=127,
        expand_nested=True,
        show_trainable=True,
        **kwargs,
    ):
        super().summary(line_length=line_length, expand_nested=expand_nested, show_trainable=show_trainable, **kwargs)

    def save(
        self,
        filepath,
        overwrite=True,
        save_format=None,
        **kwargs,
    ):
        with file_util.save_file(filepath) as path:
            super().save(filepath=path, overwrite=overwrite, save_format=save_format, **kwargs)

    def save_weights(
        self,
        filepath,
        overwrite=True,
        save_format=None,
        options=None,
    ):
        with file_util.save_file(filepath) as path:
            super().save_weights(filepath=path, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(
        self,
        filepath,
        by_name=False,
        skip_mismatch=False,
        options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

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
        optimizer,
        run_eagerly=None,
        mxp="none",
        ga_steps=None,
        gwn_config=None,
        gradn_config=None,
        **kwargs,
    ):
        optimizer = keras.optimizers.get(optimizer)
        if env_util.has_devices("TPU"):
            self.use_loss_scale = False
        else:
            self.use_loss_scale = mxp != "none" and self.dtype_policy.name == "mixed_float16"
            if self.use_loss_scale:
                logger.info("Using loss scale")  # keras auto wrap optimizer with mixed precision loss scale optimizer
        if isinstance(ga_steps, int) and ga_steps > 1:
            self.use_ga = True
            self.ga = GradientAccumulator(ga_steps=ga_steps)
            kwargs["steps_per_execution"] = 1
            logger.info(f"Using gradient accumulation with accumulate steps = {ga_steps}")
        else:
            self.use_ga = False
        self.gwn_config = gwn_config
        self.gradn = keras.regularizers.get(gradn_config) if gradn_config else None
        self.distribute_reduction_method = "mean"
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs: schemas.TrainInput, training=False):
        raise NotImplementedError()

    # -------------------------------- STEP FUNCTIONS -------------------------------------
    def apply_gwn(self) -> list:
        return []

    def remove_gwn(self, original_weights):
        pass

    def _train_step(self, data: schemas.TrainData):
        x = data[0]
        y, _ = data_util.set_length(data[1].labels, data[1].labels_length)
        sample_weight = None

        with tf.GradientTape() as tape:
            tape.watch(x.inputs)
            original_weights = self.apply_gwn()
            outputs: schemas.TrainOutput = self(x, training=True)
            tape.watch(outputs.logits)
            y_pred = outputs.logits
            y_pred, _ = data_util.set_length(y_pred, outputs.logits_length)
            self.remove_gwn(original_weights)
            tape.watch(y_pred)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

            if self.use_loss_scale:
                loss = self.optimizer.get_scaled_loss(loss)

            if self.use_ga:  # sum of gradients so the loss must be divided
                loss = loss / self.ga.total_steps

        gradients = tape.gradient(loss, self.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        if self.use_loss_scale:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        if env_util.DEBUG:
            tf.print("")
            tf.print("Outputs", outputs)

        return gradients

    def _apply_gradients(self, gradients):
        if self.gradn is not None:
            gradients = self.gradn(step=self.optimizer.iterations, gradients=gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def train_step(self, data):
        gradients = self._train_step(data)
        self._apply_gradients(gradients)
        metrics = self.get_metrics_result()
        return metrics

    def train_step_ga(self, data, prev_gradients):
        gradients = self._train_step(data)
        if prev_gradients is not None:
            gradients = self.ga.accumulate(prev_gradients, gradients)
        metrics = self.get_metrics_result()
        return metrics, gradients

    def _test_step(self, data: schemas.TrainData):
        x = data[0]
        y, _ = data_util.set_length(data[1].labels, data[1].labels_length)
        sample_weight = None

        outputs = self(x, training=False)
        y_pred, _ = data_util.set_length(outputs.logits, outputs.logits_length)

        self.compute_loss(x, y, y_pred, sample_weight)

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
        _beam_tokens = self.recognize_beam(inputs=inputs).tokens
        return {
            "_tokens": _tokens,
            "_beam_tokens": _beam_tokens,
            "_labels": y_true.labels,
        }

    # ------------------------------------ FIT ----------------------------------- #

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            outputs = self.train_step(data)
            # Ensure counter is updated only if `train_step` succeeds.
            with tf.control_dependencies(_minimum_control_deps(outputs)):
                self._train_counter.assign_add(1)
            return outputs

        if not self.run_eagerly:
            one_step_on_data = tf.function(one_step_on_data, reduce_retracing=True, jit_compile=self.jit_compile)

        @tf.autograph.experimental.do_not_convert
        def one_ga_step_on_data(data, prev_gradients):
            """Runs a single training step on a batch of data."""
            outputs, gradients = self.train_step_ga(data, prev_gradients)
            # Ensure counter is updated only if `train_step` succeeds.
            with tf.control_dependencies(_minimum_control_deps(outputs)):
                self._train_counter.assign_add(1)
            return outputs, gradients

        if not self.run_eagerly:
            one_ga_step_on_data = tf.function(one_ga_step_on_data, reduce_retracing=True, jit_compile=self.jit_compile)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single training step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(one_step_on_data, args=(data,))
            outputs = keras_util.reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs, data

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution.numpy().item()):
                outputs, data = one_step_on_iterator(iterator)
            return outputs, data

        @tf.autograph.experimental.do_not_convert
        def ga_step_in_iterator(iterator):
            data = next(iterator)
            outputs, gradients = self.distribute_strategy.run(one_ga_step_on_data, args=(data, None))
            for _ in range(1, self.ga.total_steps):
                try:
                    data = next(iterator)
                    outputs, gradients = self.distribute_strategy.run(one_ga_step_on_data, args=(data, gradients))
                except StopIteration:
                    break
            self.distribute_strategy.run(self._apply_gradients, args=(gradients,))
            outputs = keras_util.reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs, data

        if self.use_ga:
            train_function = ga_step_in_iterator
        elif self.steps_per_execution > 1:
            train_function = multi_step_on_iterator
        else:
            train_function = one_step_on_iterator

        if not self.run_eagerly:
            train_function = tf.function(train_function, reduce_retracing=True)

        def train_function_wrapper(iterator):
            outputs, data = train_function(iterator)
            loss = outputs.get("loss")
            if loss is not None:
                loss = tf_utils.sync_to_numpy_or_python_type(loss)
                if np.isnan(loss) or np.isinf(loss):
                    io_utils.print_msg("")  # empty line for newline
                    io_utils.print_msg(f"Invalid loss for batch {data}")
                    self.stop_training = True
            return outputs

        self.train_function = train_function_wrapper
        return self.train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            outputs = self.test_step(data)
            with tf.control_dependencies(_minimum_control_deps(outputs)):
                self._test_counter.assign_add(1)
            return outputs

        if not self.run_eagerly and self.jit_compile:
            one_step_on_data = tf.function(one_step_on_data, reduce_retracing=True, jit_compile=True)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single test step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(one_step_on_data, args=(data,))
            outputs = keras_util.reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution.numpy().item()):
                outputs = one_step_on_iterator(iterator)
            return outputs

        if self.steps_per_execution > 1:
            test_function = multi_step_on_iterator
        else:
            test_function = one_step_on_iterator

        if not self.run_eagerly:
            test_function = tf.function(test_function, reduce_retracing=True)

        self.test_function = test_function
        return self.test_function

    # -------------------------------- INFERENCE FUNCTIONS -------------------------------------

    def get_initial_tokens(self, batch_size=1):
        return tf.ones([batch_size, 1], dtype=tf.int32) * self.blank

    def get_initial_encoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    def get_initial_decoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

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
            )

        input_signature = schemas.PredictInput(
            inputs=tf.TensorSpec([batch_size, None], dtype=tf.float32),
            inputs_length=tf.TensorSpec([batch_size], dtype=tf.int32),
            previous_tokens=tf.TensorSpec.from_tensor(self.get_initial_tokens(batch_size)),
            previous_encoder_states=tf.TensorSpec.from_tensor(self.get_initial_encoder_states(batch_size)),
            previous_decoder_states=tf.TensorSpec.from_tensor(self.get_initial_decoder_states(batch_size)),
        )

        return tf.function(
            tflite_func,
            input_signature=[input_signature],
            jit_compile=True,
            reduce_retracing=True,
            autograph=True,
        )
