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

from ..utils import file_util, env_util


class BaseModel(tf.keras.Model):
    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None,
             save_traces=True):
        with file_util.save_file(filepath) as path:
            super().save(
                filepath=path,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces
            )

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        with file_util.save_file(filepath) as path:
            super().save_weights(
                filepath=path,
                overwrite=overwrite,
                save_format=save_format,
                options=options
            )

    def load_weights(self,
                     filepath,
                     by_name=False,
                     skip_mismatch=False,
                     options=None):
        with file_util.read_file(filepath) as path:
            super().load_weights(
                filepath=path,
                by_name=by_name,
                skip_mismatch=skip_mismatch,
                options=options
            )

    @property
    def metrics(self):
        return [self.loss_metric]

    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    def compile(self, loss, optimizer, run_eagerly=None, **kwargs):
        self.use_loss_scale = False
        if not env_util.has_tpu():
            optimizer = mxp.experimental.LossScaleOptimizer(tf.keras.optimizers.get(optimizer), "dynamic")
            self.use_loss_scale = True
        self.loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    # -------------------------------- STEP FUNCTIONS -------------------------------------

    def train_step(self, batch):
        inputs, y_true = batch
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.loss(y_true, y_pred)
            if self.use_loss_scale:
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_weights)
        if self.use_loss_scale:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        inputs, y_true = batch
        y_pred = self(inputs, training=False)
        loss = self.loss(y_true, y_pred)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

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
            beam_search_decoding = tf.map_fn(lambda _: tf.convert_to_tensor("", dtype=tf.string), labels)
        else:
            beam_search_decoding = self.recognize_beam(inputs)
        return tf.stack([labels, greedy_decoding, beam_search_decoding], axis=-1)

    def recognize(self, features, input_lengths, **kwargs):
        pass

    def recognize_beam(self, features, input_lengths, **kwargs):
        pass
