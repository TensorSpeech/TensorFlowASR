"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
from __future__ import absolute_import

import tensorflow as tf
from models.RowConv1D import RowConv1D
from models.BNRNNCell import BNLSTMCell


class DeepSpeech2:
    def __init__(self, num_conv=3, num_rnn=3, rnn_units=128):
        self.optimizer = tf.keras.optimizers.Adam
        self.num_conv = num_conv
        self.num_rnn = num_rnn
        self.rnn_units = rnn_units

    def __call__(self, features):
        layer = features
        for _ in range(self.num_conv):
            layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(41, 11),
                                           strides=(1, 2), padding="same")(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU(max_value=20)(layer)

        # combine channel dimension to features
        batch_size = tf.shape(layer)[0]
        feat_size, channel = layer.get_shape().as_list()[2:]
        layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

        # RNN layers
        for _ in range(self.num_rnn):
            layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.RNN(BNLSTMCell(self.rnn_units),
                                    return_sequences=True))(layer)

        layer = tf.keras.layers.BatchNormalization()(layer)

        return layer


class DeepSpeech2RowConv:
    def __init__(self, num_conv=3, num_rnn=3, rnn_units=256):
        self.optimizer = tf.keras.optimizers.Adam
        self.rnn_unit = rnn_units
        self.num_conv = num_conv
        self.num_rnn = num_rnn

    def __call__(self, features):
        layer = features
        for _ in range(self.num_conv):
            layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(41, 11),
                                           strides=(1, 2), padding="same")(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU(max_value=20)(layer)

        # combine channel dimension to features
        batch_size = tf.shape(layer)[0]
        feat_size, channel = layer.get_shape().as_list()[2:]
        layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

        # RNN layers
        for _ in range(self.num_rnn):
            layer = tf.keras.layers.LSTM(self.rnn_unit, activation='tanh',
                                         recurrent_activation='sigmoid',
                                         return_sequences=True,
                                         recurrent_dropout=0,
                                         unroll=False, use_bias=True)(layer)
            layer = RowConv1D(filters=self.rnn_unit, future_context=2, strides=1,
                              padding="same")(layer)

        layer = tf.keras.layers.BatchNormalization()(layer)

        return layer


class DeepSpeech2RowConvBNRNN:
    def __init__(self, num_conv=3, num_rnn=3, rnn_units=256):
        self.optimizer = tf.keras.optimizers.Adam
        self.rnn_unit = rnn_units
        self.num_conv = num_conv
        self.num_rnn = num_rnn

    def __call__(self, features):
        layer = features
        for _ in range(self.num_conv):
            layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(41, 11),
                                           strides=(1, 2), padding="same")(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU(max_value=20)(layer)

        # combine channel dimension to features
        batch_size = tf.shape(layer)[0]
        feat_size, channel = layer.get_shape().as_list()[2:]
        layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

        # RNN layers
        for _ in range(self.num_rnn):
            layer = tf.keras.layers.RNN(
                BNLSTMCell(self.rnn_unit, activation='tanh',
                           recurrent_activation='sigmoid', use_bias=True),
                return_sequences=True, unroll=False)(layer)
            layer = RowConv1D(filters=self.rnn_unit, future_context=2, strides=1,
                              padding="same")(layer)

        layer = tf.keras.layers.BatchNormalization()(layer)

        return layer
