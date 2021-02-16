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

import tensorflow as tf
from tensorflow.keras import mixed_precision as mxp

from ..transducer import Transducer as BaseTransducer
from ...utils.utils import get_reduced_length
from ...losses.keras.rnnt_losses import RnntLoss


class Transducer(BaseTransducer):
    """ Keras Transducer Model Warper """

    def _build(self, input_shape, prediction_shape=[None], batch_size=None):
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        input_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        pred = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        pred_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self({
            "input": inputs,
            "input_length": input_length,
            "prediction": pred,
            "prediction_length": pred_length
        }, training=False)

    def call(self, inputs, training=False, **kwargs):
        features = inputs["input"]
        prediction = inputs["prediction"]
        prediction_length = inputs["prediction_length"]
        enc = self.encoder(features, training=training, **kwargs)
        pred = self.predict_net([prediction, prediction_length], training=training, **kwargs)
        outputs = self.joint_net([enc, pred], training=training, **kwargs)
        return {
            "logit": outputs,
            "logit_length": get_reduced_length(inputs["input_length"], self.time_reduction_factor)
        }

    def compile(self, optimizer, global_batch_size, blank=0, use_loss_scale=False,
                loss_weights=None, weighted_metrics=None, run_eagerly=None, **kwargs):
        loss = RnntLoss(blank=blank, global_batch_size=global_batch_size)
        self.use_loss_scale = use_loss_scale
        if self.use_loss_scale:
            optimizer = mxp.experimental.LossScaleOptimizer(tf.keras.optimizers.get(optimizer), 'dynamic')
        super(Transducer, self).compile(
            optimizer=optimizer, loss=loss,
            loss_weights=loss_weights, weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs
        )

    def train_step(self, batch):
        x, y_true = batch
        with tf.GradientTape() as tape:
            y_pred = self({
                "input": x["input"],
                "input_length": x["input_length"],
                "prediction": x["prediction"],
                "prediction_length": x["prediction_length"],
            }, training=True)
            loss = self.loss(y_true, y_pred)
            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_loss_scale:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"rnnt_loss": loss}

    def test_step(self, batch):
        x, y_true = batch
        y_pred = self({
            "input": x["input"],
            "input_length": x["input_length"],
            "prediction": x["prediction"],
            "prediction_length": x["prediction_length"],
        }, training=False)
        loss = self.loss(y_true, y_pred)
        return {"rnnt_loss": loss}
