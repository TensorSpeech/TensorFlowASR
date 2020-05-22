from __future__ import absolute_import

import tensorflow as tf


# https://arxiv.org/abs/1510.01378
class SequenceBatchNorm(tf.keras.layers.Layer):
    def __init__(self, name, time_major=False, **kwargs):
        super(SequenceBatchNorm, self).__init__(name=name, **kwargs)
        self.time_major = time_major

    def build(self, input_shape):
        self.beta = self.add_weight(shape=[input_shape[-1]],
                                    name='beta', initializer='zeros',
                                    regularizer=None, constraint=None, trainable=True)
        self.gamma = self.add_weight(shape=[input_shape[-1]],
                                     name='gamma', initializer='ones',
                                     regularizer=None, constraint=None, trainable=True)

    def call(self, inputs, **kwargs):
        if self.time_major:
            total_padded_frames = tf.cast(tf.shape(inputs)[0], tf.float32)
            batch_size = tf.cast(tf.shape(inputs)[1], tf.float32)
        else:
            total_padded_frames = tf.cast(tf.shape(inputs)[1], tf.float32)
            batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
        mean, variance = tf.nn.moments(inputs, axes=[0, 1], keepdims=False)
        total_unpadded_frames_batch = tf.math.count_nonzero(inputs, axis=[0, 1], keepdims=False, dtype=tf.float32)
        mean = (mean * total_padded_frames * batch_size) / total_unpadded_frames_batch
        variance = (variance * total_padded_frames * batch_size) / total_unpadded_frames_batch
        return tf.nn.batch_normalization(inputs, mean=mean, variance=variance,
                                         offset=self.beta, scale=self.gamma, variance_epsilon=tf.keras.backend.epsilon())

    def get_config(self):
        config = super(SequenceBatchNorm, self).get_config()
        config.update({
            "time_major": self.time_major
        })
        return config

    def from_config(self, config):
        return self(**config)
