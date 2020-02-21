from __future__ import absolute_import

import tensorflow as tf
import functools


class DeepSpeech2:
    def __init__(self):
        self.clipped_relu = functools.partial(tf.keras.activations.relu, max_value=20)
        self.optimizer = tf.keras.optimizers.Adam

    def __call__(self, features):
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(41, 11), strides=(1, 2), padding="same",
                                       activation=self.clipped_relu)(features)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(41, 11), strides=(1, 2), padding="same",
                                       activation=self.clipped_relu)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(41, 11), strides=(1, 2), padding="same",
                                       activation=self.clipped_relu)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)

        # combine channel dimension to features
        batch_size = tf.shape(layer)[0]
        feat_size, channel = layer.get_shape().as_list()[2:]
        layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

        # RNN layers
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(layer)

        return layer
