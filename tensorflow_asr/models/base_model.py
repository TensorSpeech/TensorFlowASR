# pylint: disable=attribute-defined-outside-init
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

import tensorflow as tf

from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.optimizers.accumulation import GradientAccumulator
from tensorflow_asr.utils import env_util, file_util

logger = tf.get_logger()


class BaseModel(tf.keras.Model):
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

    @property
    def metrics(self):
        if not hasattr(self, "_tfasr_metrics"):
            self._tfasr_metrics = {}
        return list(self._tfasr_metrics.values())

    def reset_metrics(self):
        super().reset_metrics()
        self.reset_states()  # reset all stateful states also

    def add_custom_metric(self, metric: tf.keras.metrics.Metric):
        if not hasattr(self, "_tfasr_metrics"):
            self._tfasr_metrics = {}
        self._tfasr_metrics[metric.name] = metric

    def make(self, *args, **kwargs):
        """Custom function for building model (uses self.build so cannot overwrite that function)"""
        raise NotImplementedError()

    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        mxp="none",
        ga_steps=None,
        apply_gwn_config=None,
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
            self.ga = GradientAccumulator(ga_steps=ga_steps, trainable_variables=self.trainable_variables)
            logger.info(f"Using gradient accumulation with accumulate steps = {ga_steps}")
        else:
            self.use_ga = False
        self.apply_gwn_config = apply_gwn_config
        self.add_custom_metric(metric=tf.keras.metrics.Mean(name="loss"))
        self.distribute_reduction_method = "sum"
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def add_featurizers(self, speech_featurizer: SpeechFeaturizer, text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    # -------------------------------- STEP FUNCTIONS -------------------------------------
    def apply_gwn(self) -> list:
        return []

    def remove_gwn(self, original_weights):
        pass

    def _get_global_batch_size(self, y_pred):
        global_batch_size = tf.shape(y_pred["logits"])[0] * self.distribute_strategy.num_replicas_in_sync
        return global_batch_size

    def train_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of training data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric

        """
        inputs, y_true = batch

        with tf.GradientTape() as tape:
            original_weights = self.apply_gwn()
            y_pred = self(inputs, training=True)
            self.remove_gwn(original_weights)
            tape.watch(y_pred["logits"])
            per_sample_loss = self.loss(y_true=y_true, y_pred=y_pred)
            global_batch_size = self._get_global_batch_size(y_pred)
            loss = tf.nn.compute_average_loss(per_sample_loss, global_batch_size=global_batch_size)
            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.use_loss_scale:
            gradients = tape.gradient(scaled_loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        if self.use_ga:  # perform gradient accumulation
            self.ga.accumulate(gradients=gradients)
            self.optimizer.apply_gradients(zip(self.ga.gradients, self.trainable_variables))
            tf.cond(self.ga.is_apply_step, self.ga.reset, lambda: None)
        else:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._tfasr_metrics["loss"].update_state(per_sample_loss)
        result = {m.name: m.result() / tf.distribute.get_strategy().num_replicas_in_sync for m in self.metrics}
        return result

    def test_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]: a batch of validation data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric prefixed with "val_"

        """
        inputs, y_true = batch
        y_pred = self(inputs, training=False)
        per_sample_loss = self.loss(y_true=y_true, y_pred=y_pred)
        # global_batch_size = self._get_global_batch_size(y_pred)
        # loss = tf.nn.compute_average_loss(per_sample_loss, global_batch_size=global_batch_size)
        self._tfasr_metrics["loss"].update_state(per_sample_loss)
        return {m.name: m.result() / tf.distribute.get_strategy().num_replicas_in_sync for m in self.metrics}

    def predict_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of testing data

        Returns:
            [tf.Tensor]: stacked tensor of shape [B, 3] with each row is the text [truth, greedy, beam_search]
        """
        inputs, y_true = batch
        labels = self.text_featurizer.iextract(y_true["labels"])
        greedy_decoding = self.recognize(inputs)
        if self.text_featurizer.decoder_config.beam_width == 0:
            beam_search_decoding = tf.tile(tf.expand_dims(tf.convert_to_tensor("", tf.string), 0), [tf.shape(labels)[0]])
        else:
            beam_search_decoding = self.recognize_beam(inputs)
        return tf.stack([labels, greedy_decoding, beam_search_decoding], axis=-1)

    # -------------------------------- INFERENCE FUNCTIONS -------------------------------------

    def recognize(self, *args, **kwargs):
        """Greedy decoding function that used in self.predict_step"""
        raise NotImplementedError()

    def recognize_beam(self, *args, **kwargs):
        """Beam search decoding function that used in self.predict_step"""
        raise NotImplementedError()

    # ---------------------------------- TFLITE ---------------------------------- #

    def make_tflite_function(self, *args, **kwargs):
        pass
