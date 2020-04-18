"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
from __future__ import absolute_import

import tensorflow as tf
from models.deepspeech2.RowConv1D import RowConv1D
from models.deepspeech2.BNRNNCell import BNLSTMCell


class DeepSpeech2:
  def __init__(self, num_conv=3, num_rnn=3, rnn_units=128,
               filters=32, kernel_size=(21, 11),
               is_bidirectional=False, is_rowconv=False):
    self.optimizer = tf.keras.optimizers.Adam
    self.num_conv = num_conv
    self.num_rnn = num_rnn
    self.rnn_units = rnn_units
    self.filters = filters
    self.kernel_size = kernel_size
    self.is_bidirectional = is_bidirectional
    self.is_rowconv = is_rowconv

  def __call__(self, features, streaming=False):
    if streaming:
      raise ValueError("This model cannot be used in streaming mode")
    layer = features
    for i in range(self.num_conv):
      layer = tf.keras.layers.Conv2D(
        filters=self.filters, kernel_size=self.kernel_size,
        strides=(1, 2), padding="same", name=f"cnn_{i}")(layer)
      layer = tf.keras.layers.BatchNormalization(name=f"bn_cnn_{i}")(layer)
      layer = tf.keras.layers.ReLU(max_value=20, name=f"relu_cnn_{i}")(layer)

    # combine channel dimension to features
    batch_size = tf.shape(layer)[0]
    feat_size, channel = layer.get_shape().as_list()[2:]
    layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

    # Convert to time_major
    layer = tf.transpose(layer, [1, 0, 2])

    # RNN layers
    for i in range(self.num_rnn):
      if self.is_bidirectional:
        layer = tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(self.rnn_units, dropout=0.2,
                               return_sequences=True, unroll=False,
                               recurrent_dropout=0.0, use_bias=True,
                               time_major=True, stateful=False),
          name=f"bilstm_{i}")(layer)
      else:
        layer = tf.keras.layers.LSTM(self.rnn_units, dropout=0.2,
                                     return_sequences=True, unroll=False,
                                     recurrent_dropout=0.0, use_bias=True,
                                     time_major=True, stateful=streaming,
                                     name=f"lstm_{i}")(layer)

    # Convert to batch_major
    layer = tf.transpose(layer, [1, 0, 2])

    layer = tf.keras.layers.Dense(units=1024,
                                  name="pre_fully_connected",
                                  use_bias=True)(layer)
    layer = tf.keras.layers.ReLU(max_value=20, name=f"relu_pre_fc_{i}")(layer)

    return layer
