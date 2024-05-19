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

import copy
import importlib

import tensorflow as tf
from keras import callbacks as callbacks_module
from keras.optimizers import Optimizer
from tensorflow.python.eager import context  # pylint: disable=no-name-in-module

from tensorflow_asr import schemas
from tensorflow_asr.models.layers.feature_extraction import FeatureExtraction
from tensorflow_asr.optimizers.accumulation import GradientAccumulator
from tensorflow_asr.tokenizers import Tokenizer
from tensorflow_asr.utils import data_util, env_util, file_util, math_util, shape_util

base_layer = importlib.import_module(f"{env_util.KERAS_SRC}.engine.base_layer")
data_adapter = importlib.import_module(f"{env_util.KERAS_SRC}.engine.data_adapter")
training_utils = importlib.import_module(f"{env_util.KERAS_SRC}.engine.training_utils")

tf_utils = importlib.import_module(f"{env_util.KERAS_SRC}.utils.tf_utils")
version_utils = importlib.import_module(f"{env_util.KERAS_SRC}.utils.version_utils")

_disallow_inside_tf_function = importlib.import_module(f"{env_util.KERAS_SRC}.engine.training")._disallow_inside_tf_function
_get_verbosity = importlib.import_module(f"{env_util.KERAS_SRC}.engine.training")._get_verbosity
_minimum_control_deps = importlib.import_module(f"{env_util.KERAS_SRC}.engine.training")._minimum_control_deps
reduce_per_replica = importlib.import_module(f"{env_util.KERAS_SRC}.engine.training").reduce_per_replica

logger = tf.get_logger()


class BaseModel(tf.keras.Model):
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
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        with file_util.save_file(filepath) as path:
            super().save(
                filepath=path,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

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

    def add_custom_metric(self, metric: tf.keras.metrics.Metric):
        if not hasattr(self, "_tfasr_metrics"):
            self._tfasr_metrics = {}
        self._tfasr_metrics[metric.name] = metric

    def make(self, input_shape=[None], prediction_shape=[None], batch_size=None, caching=None, **kwargs) -> schemas.TrainOutput:
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
        signals = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        signals_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self._per_replica_batch_size = int(batch_size / self.distribute_strategy.num_replicas_in_sync)
        self._batch_size = batch_size
        outputs = self(
            schemas.TrainInput(
                inputs=signals,
                inputs_length=signals_length,
                predictions=predictions,
                predictions_length=predictions_length,
                caching=caching,
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
        optimizer = tf.keras.optimizers.get(optimizer)
        if env_util.has_devices("TPU"):
            self.use_loss_scale = False
        else:
            self.use_loss_scale = mxp != "none"
            if self.use_loss_scale:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                logger.info("Using loss scale")
        if isinstance(ga_steps, int) and ga_steps > 1:
            self.use_ga = True
            self.ga = GradientAccumulator(ga_steps=ga_steps, model=self)
            logger.info(f"Using gradient accumulation with accumulate steps = {ga_steps}")
        else:
            self.use_ga = False
        self.gwn_config = gwn_config
        self.gradn = tf.keras.regularizers.get(gradn_config) if gradn_config else None
        self.distribute_reduction_method = "sum"
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs: schemas.TrainInput, training=False):
        raise NotImplementedError()

    # -------------------------------- STEP FUNCTIONS -------------------------------------
    def apply_gwn(self) -> list:
        return []

    def remove_gwn(self, original_weights):
        pass

    def _get_global_batch_size(self, y_pred):
        global_batch_size = tf.shape(y_pred["logits"])[0] * self.distribute_strategy.num_replicas_in_sync
        return global_batch_size

    def _validate_and_get_metrics_result(self, logs):
        logs = super()._validate_and_get_metrics_result(logs)
        if "predictions" in logs:
            del logs["predictions"]
        return logs

    def _train_step(self, data, caching=None):
        x = data[0]
        if caching is not None:
            x["caching"] = caching
        y, _ = data_util.attach_length_to_data(data[1]["labels"], data[1]["labels_length"])
        sample_weight = None

        with tf.GradientTape() as tape:
            tape.watch(x["inputs"])
            original_weights = self.apply_gwn()
            outputs = self(x, training=True)
            tape.watch(outputs["logits"])
            y_pred, caching = outputs["logits"], outputs.get("caching")
            y_pred, _ = data_util.attach_length_to_data(y_pred, outputs["logits_length"])
            self.remove_gwn(original_weights)
            tape.watch(y_pred)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

            if self.use_ga:  # sum of gradients so the loss must be divided
                loss = loss / self.ga.total_steps

            if self.use_loss_scale:
                loss = self.optimizer.get_scaled_loss(loss)
                gradients = tape.gradient(loss, self.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(gradients)
            else:
                gradients = tape.gradient(loss, self.trainable_variables)

        return gradients, caching

    def train_step(self, data, caching=None):
        if not self.use_ga:
            gradients, caching = self._train_step(data, caching=caching)
        else:
            if caching is None:  # separate 2 cases for tf.while_loop to avoid errors
                for i in tf.range(self.ga.total_steps):
                    per_ga_step_data = tf.nest.map_structure(
                        lambda x: math_util.slice_batch_tensor(x, index=i, batch_size=self._per_replica_batch_size), data
                    )
                    per_ga_gradients, _ = self._train_step(per_ga_step_data)
                    self.ga.accumulate(per_ga_gradients)
            else:
                for i in tf.range(self.ga.total_steps):
                    per_ga_step_data = tf.nest.map_structure(
                        lambda x: math_util.slice_batch_tensor(x, index=i, batch_size=self._per_replica_batch_size), data
                    )
                    per_ga_gradients, caching = self._train_step(per_ga_step_data, caching=caching)
                    self.ga.accumulate(per_ga_gradients)
            gradients = self.ga.gradients
        if self.gradn is not None:
            gradients = self.gradn(step=self.optimizer.iterations, gradients=gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        if self.use_ga:
            self.ga.reset()
        metrics = self.get_metrics_result()
        metrics = tf.nest.map_structure(lambda x: x / self.distribute_strategy.num_replicas_in_sync, metrics)
        return metrics, caching

    def _test_step(self, data):
        x = data[0]
        y, _ = data_util.attach_length_to_data(data[1]["labels"], data[1]["labels_length"])
        sample_weight = None

        outputs = self(x, training=False)
        y_pred, _ = data_util.attach_length_to_data(outputs["logits"], outputs["logits_length"])

        self.compute_loss(x, y, y_pred, sample_weight)

    def test_step(self, data):
        if not self.use_ga:
            self._test_step(data)
        else:
            for i in tf.range(self.ga.total_steps):
                per_ga_step_data = tf.nest.map_structure(
                    lambda x: math_util.slice_batch_tensor(x, index=i, batch_size=self._per_replica_batch_size), data
                )
                self._test_step(per_ga_step_data)
        metrics = self.get_metrics_result()
        metrics = tf.nest.map_structure(lambda x: x / self.distribute_strategy.num_replicas_in_sync, metrics)
        return metrics

    def predict_step(self, data):
        x, y_true = data
        inputs = schemas.PredictInput(
            inputs=x["inputs"],
            inputs_length=x["inputs_length"],
            previous_tokens=self.get_initial_tokens(),
            previous_encoder_states=self.get_initial_encoder_states(),
            previous_decoder_states=self.get_initial_decoder_states(),
        )
        _tokens = self.recognize(inputs=inputs).tokens
        _beam_tokens = self.recognize_beam(inputs=inputs).tokens
        return {
            "_tokens": _tokens,
            "_beam_tokens": _beam_tokens,
            "_labels": y_true["labels"],
        }

    # ------------------------------------ FIT ----------------------------------- #

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        def step_function(model, iterator, caching):
            """Runs a single training step."""

            def run_step(data, caching):
                outputs, caching = model.train_step(data, caching)
                # Ensure counter is updated only if `train_step` succeeds.
                with tf.control_dependencies(_minimum_control_deps(outputs)):
                    model._train_counter.assign_add(1)
                return outputs, caching

            run_step = tf.function(run_step, jit_compile=self.jit_compile, reduce_retracing=True)

            data = next(iterator)
            outputs, caching = model.distribute_strategy.run(run_step, args=(data, caching))
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs, caching

        # Special case if steps_per_execution is one.
        if self._steps_per_execution is None or self._steps_per_execution.numpy().item() == 1:

            def train_function(iterator, caching):
                """Runs a training execution with a single step."""
                return step_function(self, iterator, caching)

            if not self.run_eagerly:
                train_function = tf.function(train_function, reduce_retracing=True)
                self.train_tf_function = train_function

            if self._cluster_coordinator:
                self.train_function = lambda it: self._cluster_coordinator.schedule(train_function, args=(it,))
            else:
                self.train_function = train_function

        # If we're using a coordinator, use the value of
        # self._steps_per_execution at the time the function is
        # called/scheduled, and not when it is actually executed.
        elif self._cluster_coordinator:

            def train_function(iterator, caching, steps_per_execution):
                """Runs a training execution with multiple steps."""
                for _ in tf.range(steps_per_execution):
                    outputs, caching = step_function(self, iterator, caching)
                return outputs, caching

            if not self.run_eagerly:
                train_function = tf.function(train_function, reduce_retracing=True)
                self.train_tf_function = train_function
            # fmt: off
            self.train_function = lambda it, cache: self._cluster_coordinator.schedule(
                train_function, args=(it, cache, self._steps_per_execution.value())
            )  # pylint: disable=line-too-long
            # fmt: on
        else:

            def train_function(iterator, caching):
                """Runs a training execution with multiple steps."""
                for _ in tf.range(self._steps_per_execution):
                    outputs, caching = step_function(self, iterator, caching)
                return outputs, caching

            if not self.run_eagerly:
                train_function = tf.function(train_function, reduce_retracing=True)
                self.train_tf_function = train_function
            self.train_function = train_function

        return self.train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def step_function(model, iterator):
            """Runs a single evaluation step."""

            def run_step(data):
                outputs = model.test_step(data)
                # Ensure counter is updated only if `test_step` succeeds.
                with tf.control_dependencies(_minimum_control_deps(outputs)):
                    model._test_counter.assign_add(1)
                return outputs

            run_step = tf.function(run_step, jit_compile=self.jit_compile, reduce_retracing=True)

            data = next(iterator)
            outputs = model.distribute_strategy.run(run_step, args=(data,))
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        # Special case if steps_per_execution is one.
        if self._steps_per_execution is None or self._steps_per_execution.numpy().item() == 1:

            def test_function(iterator):
                """Runs a test execution with a single step."""
                return step_function(self, iterator)

            if not self.run_eagerly:
                test_function = tf.function(test_function, reduce_retracing=True)

            if self._cluster_coordinator:
                self.test_function = lambda it: self._cluster_coordinator.schedule(test_function, args=(it,))
            else:
                self.test_function = test_function

        # If we're using a coordinator, use the value of
        # self._steps_per_execution at the time the function is
        # called/scheduled, and not when it is actually executed.
        elif self._cluster_coordinator:

            def test_function(iterator, steps_per_execution):
                """Runs a test execution with multiple steps."""
                for _ in tf.range(steps_per_execution):
                    outputs = step_function(self, iterator)
                return outputs

            if not self.run_eagerly:
                test_function = tf.function(test_function, reduce_retracing=True)

            self.test_function = lambda it: self._cluster_coordinator.schedule(test_function, args=(it, self._steps_per_execution.value()))
        else:

            def test_function(iterator):
                """Runs a test execution with multiple steps."""
                for _ in tf.range(self._steps_per_execution):
                    outputs = step_function(self, iterator)
                return outputs

            if not self.run_eagerly:
                test_function = tf.function(test_function, reduce_retracing=True)
            self.test_function = test_function

        return self.test_function

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        base_layer.keras_api_gauge.get_cell("fit").set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph("Model", "fit")
        self._assert_compile_was_called()
        self._check_call_args("fit")
        _disallow_inside_tf_function("fit")

        verbose = _get_verbosity(verbose, self.distribute_strategy)

        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for `Tensor` and `NumPy` input.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter.train_validation_split((x, y, sample_weight), validation_split=validation_split)

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter.unpack_x_y_sample_weight(validation_data)

        if self.distribute_strategy._should_use_with_coordinator:
            self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(self.distribute_strategy)

        with self.distribute_strategy.scope(), training_utils.RespectCompiledTrainableState(self):  # noqa: E501
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.get_data_handler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution,
            )

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps,
                )

            self.stop_training = False
            self.train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            steps_per_epoch_inferred = steps_per_epoch or data_handler.inferred_steps
            (
                data_handler._initial_epoch,
                data_handler._initial_step,
            ) = self._maybe_load_initial_counters_from_ckpt(steps_per_epoch_inferred, initial_epoch)
            logs = None
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                caching = (
                    self.distribute_strategy.experimental_distribute_values_from_function(lambda ctx: self.reset_caching())
                    if hasattr(self, "reset_caching")
                    else None
                )
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with tf.profiler.experimental.Trace(
                            "train",
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=batch_size,
                            _r=1,
                        ):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs, caching = self.train_function(iterator, caching=caching)
                            if data_handler.should_sync:
                                context.async_wait()
                            # No error, now safe to assign to logs.
                            logs = tmp_logs
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                            if self.stop_training:
                                break

                logs = tf_utils.sync_to_numpy_or_python_type(logs)
                if logs is None:
                    raise ValueError(
                        "Unexpected result of `train_function` "
                        "(Empty logs). Please use "
                        "`Model.compile(..., run_eagerly=True)`, or "
                        "`tf.config.run_functions_eagerly(True)` for more "
                        "information of where went wrong, or file a "
                        "issue/bug to `tf.keras`."
                    )
                # Override with model metrics instead of last step logs
                logs = self._validate_and_get_metrics_result(logs)
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, "_eval_data_handler", None) is None:
                        self._eval_data_handler = data_adapter.get_data_handler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution,
                        )
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True,
                        _use_cached_eval_dataset=True,
                    )
                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            if isinstance(self.optimizer, Optimizer) and epochs > 0:
                self.optimizer.finalize_variable_values(self.trainable_variables)

            # If eval data_handler exists, delete it after all epochs are done.
            if getattr(self, "_eval_data_handler", None) is not None:
                del self._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return self.history

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
