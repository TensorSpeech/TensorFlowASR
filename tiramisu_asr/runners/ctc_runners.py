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
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from ..featurizers.speech_featurizers import SpeechFeaturizer, TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..losses.ctc_losses import ctc_loss
from .base_runners import BaseTrainer


class CTCTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: TFSpeechFeaturizer or SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy: tf.distribute.Strategy = None):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision
        super(CTCTrainer, self).__init__(config=config, strategy=strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
        }

    def save_model_weights(self):
        with self.strategy.scope():
            self.model.save_weights(os.path.join(self.config["outdir"], "latest.h5"))

    def _train_step(self, batch):
        _, features, input_length, labels, label_length, _ = batch

        with tf.GradientTape() as tape:
            y_pred = self.model(features, training=True)
            tape.watch(y_pred)
            per_train_loss = ctc_loss(
                y_true=labels, y_pred=y_pred,
                input_length=(input_length // self.model.time_reduction_factor),
                label_length=label_length,
                blank=self.text_featurizer.blank
            )
            train_loss = tf.nn.compute_average_loss(per_train_loss,
                                                    global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["ctc_loss"].update_state(per_train_loss)

    def _eval_step(self, batch):
        _, features, input_length, labels, label_length, _ = batch

        logits = self.model(features, training=False)

        per_eval_loss = ctc_loss(
            y_true=labels, y_pred=logits,
            input_length=(input_length // self.model.time_reduction_factor),
            label_length=label_length,
            blank=self.text_featurizer.blank
        )

        # Update metrics
        self.eval_metrics["ctc_loss"].update_state(per_eval_loss)

    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.model = model
            self.optimizer = tf.keras.optimizers.get(optimizer)
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.create_checkpoint_manager(max_to_keep, model=self.model, optimizer=self.optimizer)
