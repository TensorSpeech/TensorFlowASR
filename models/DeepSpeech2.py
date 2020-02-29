from __future__ import absolute_import

import tensorflow as tf
import functools
from models.RowConv1D import RowConv1D


class DeepSpeech2:
    def __init__(self):
        self.clipped_relu = functools.partial(tf.keras.activations.relu, max_value=20)
        self.optimizer = tf.keras.optimizers.Adam

    def __call__(self, features):
        layer = features
        for i in range(2):
            layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(21, 11),
                                           strides=(1, 2), padding="same")(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Activation(activation=self.clipped_relu)(layer)
            layer = tf.keras.layers.Dropout(0.2)(layer)

        # combine channel dimension to features
        batch_size = tf.shape(layer)[0]
        feat_size, channel = layer.get_shape().as_list()[2:]
        layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

        # RNN layers
        for i in range(2):
            layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True,
                                     recurrent_dropout=0.2))(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)

        return layer


class DeepSpeech2RowConv:
    def __init__(self):
        self.clipped_relu = functools.partial(tf.keras.activations.relu, max_value=20)
        self.optimizer = tf.keras.optimizers.Adam
        self.rnn_unit = 128

    def __call__(self, features):
        layer = features
        for i in range(2):
            layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(21, 11),
                                           strides=(1, 2), padding="same")(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Activation(activation=self.clipped_relu)(layer)
            layer = tf.keras.layers.Dropout(0.2)(layer)

        # combine channel dimension to features
        batch_size = tf.shape(layer)[0]
        feat_size, channel = layer.get_shape().as_list()[2:]
        layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

        # RNN layers
        for i in range(2):
            layer = tf.keras.layers.LSTM(self.rnn_unit, return_sequences=True,
                                         recurrent_dropout=0.2)(layer)
            layer = RowConv1D(filters=self.rnn_unit, future_context=2, strides=1,
                              padding="same")(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)

        return layer
