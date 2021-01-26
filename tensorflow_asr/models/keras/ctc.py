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

    def compile(self, optimizer, global_batch_size, blank=0,
                loss_weights=None, weighted_metrics=None, run_eagerly=None, **kwargs):
        loss = CtcLoss(blank=blank, global_batch_size=global_batch_size)
        optimizer_with_scale = mxp.experimental.LossScaleOptimizer(tf.keras.optimizers.get(optimizer), 'dynamic')
        super(CtcModel, self).compile(
            optimizer=optimizer_with_scale, loss=loss,
            loss_weights=loss_weights, weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs
        )

    def train_step(self, batch):
        x, y_true = batch
        with tf.GradientTape() as tape:
            logit = self(x['input'], training=True)
            y_pred = {
                'logit': logit,
                'logit_length': get_reduced_length(x['input_length'], self.time_reduction_factor)
            }
            loss = self.loss(y_true, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"train_ctc_loss": loss}

    def test_step(self, batch):
        x, y_true = batch
        logit = self(x, training=False)
        y_pred = {
            'logit': logit,
            'logit_length': get_reduced_length(x['input_length'], self.time_reduction_factor)
        }
        loss = self.loss(y_true, y_pred)
        return {"val_ctc_loss": loss}
