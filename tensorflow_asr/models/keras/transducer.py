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
from tensorflow.python.keras.engine import data_adapter

from ..transducer import Transducer as BaseTransducer
from ...utils.utils import get_reduced_length


class Transducer(BaseTransducer):
    """ Keras Transducer Model Warper """

    def _build(self, input_shape):
        features = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        input_length = tf.keras.Input(shape=[], dtype=tf.int32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        pred_length = tf.keras.Input(shape=[], dtype=tf.int32)
        self({
            "input": features,
            "input_length": input_length,
            "prediction": pred,
            "prediction_length": pred_length
        }, training=True)

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

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
