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
from tensorflow.keras import mixed_precision as mxp

from ..ctc import CtcModel as BaseCtcModel
from ...utils.utils import get_reduced_length
from ...losses.keras.ctc_losses import CtcLoss


class CtcModel(BaseCtcModel):
    """ Keras CTC Model Warper """
    @property
    def metrics(self):
        return [self.loss_metric]

    def compile(self, optimizer, global_batch_size, blank=0, use_loss_scale=False, run_eagerly=None, **kwargs):
        loss = CtcLoss(blank=blank, global_batch_size=global_batch_size)
        self.use_loss_scale = use_loss_scale
        if self.use_loss_scale:
            optimizer = mxp.experimental.LossScaleOptimizer(tf.keras.optimizers.get(optimizer), "dynamic")
        self.loss_metric = tf.keras.metrics.Mean(name="ctc_loss", dtype=tf.float32)
        super(CtcModel, self).compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def train_step(self, batch):
        x, y_true = batch
        with tf.GradientTape() as tape:
            logit = self(x["input"], training=True)
            y_pred = {
                "logit": logit,
                "logit_length": get_reduced_length(x["input_length"], self.time_reduction_factor)
            }
            loss = self.loss(y_true, y_pred)
            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_loss_scale:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        x, y_true = batch
        logit = self(x["input"], training=False)
        y_pred = {
            "logit": logit,
            "logit_length": get_reduced_length(x["input_length"], self.time_reduction_factor)
        }
        loss = self.loss(y_true, y_pred)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}
