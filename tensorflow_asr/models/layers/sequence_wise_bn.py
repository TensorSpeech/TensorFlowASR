# pylint:disable=attribute-defined-outside-init
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
import tensorflow as tf


# https://arxiv.org/abs/1510.01378
class SequenceBatchNorm(tf.keras.layers.Layer):
    def __init__(self, name, time_major=False, gamma_regularizer=None, beta_regularizer=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time_major = time_major
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)

    def build(
        self,
        input_shape,
    ):
        self.beta = self.add_weight(
            shape=[input_shape[-1]],
            name="beta",
            initializer="zeros",
            regularizer=self.beta_regularizer,
            constraint=None,
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.gamma = self.add_weight(
            shape=[input_shape[-1]],
            name="gamma",
            initializer="ones",
            regularizer=self.gamma_regularizer,
            constraint=None,
            trainable=True,
            dtype=self.variable_dtype,
        )

    def call(
        self,
        inputs,
        **kwargs,
    ):
        mean, variance = tf.nn.moments(inputs, axes=[0, 1], keepdims=False)
        if self.time_major:
            total_padded_frames = tf.cast(tf.shape(inputs)[0], tf.keras.backend.dtype(mean))
            batch_size = tf.cast(tf.shape(inputs)[1], tf.keras.backend.dtype(mean))
        else:
            total_padded_frames = tf.cast(tf.shape(inputs)[1], tf.keras.backend.dtype(mean))
            batch_size = tf.cast(tf.shape(inputs)[0], tf.keras.backend.dtype(mean))
        total_unpadded_frames_batch = tf.math.count_nonzero(inputs, axis=[0, 1], keepdims=False, dtype=tf.keras.backend.dtype(mean))
        mean = (mean * total_padded_frames * batch_size) / total_unpadded_frames_batch
        variance = (variance * total_padded_frames * batch_size) / total_unpadded_frames_batch
        return tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=tf.keras.backend.epsilon(),
        )
