"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
from __future__ import absolute_import

import tensorflow as tf
from models.deepspeech2.RowConv1D import RowConv1D
from models.deepspeech2.SequenceBatchNorm import SequenceBatchNorm


class DeepSpeech2:
  def __init__(self, num_conv=3, num_rnn=3, rnn_units=128,
               filters=(32, 32, 32), kernel_size=(31, 11),
               is_bidirectional=False, is_rowconv=False, pre_fc_units=1024):
    self.optimizer = tf.keras.optimizers.Adam
    self.num_conv = num_conv
    self.num_rnn = num_rnn
    self.rnn_units = rnn_units
    self.filters = filters
    self.kernel_size = kernel_size
    self.is_bidirectional = is_bidirectional
    self.is_rowconv = is_rowconv
    self.pre_fc_units = pre_fc_units

  def __call__(self, features, streaming=False):
    layer = features
    for i in range(self.num_conv):
      layer = tf.keras.layers.Conv2D(
        filters=self.filters[i] if isinstance(self.filters, list) else self.filters,
        kernel_size=self.kernel_size,
        strides=(1, 2), padding="same", name=f"cnn_{i}")(layer)
      layer = tf.keras.layers.BatchNormalization(axis=-1, name=f"bn_cnn_{i}")(layer)
      layer = tf.keras.layers.ReLU(name=f"relu_cnn_{i}")(layer)

    # combine channel dimension to features
    batch_size = tf.shape(layer)[0]
    feat_size, channel = layer.get_shape().as_list()[2:]
    layer = tf.reshape(layer, [batch_size, -1, feat_size * channel])

    # Convert to time_major only for bi_directional
    if self.is_bidirectional:
      layer = tf.transpose(layer, [1, 0, 2])

    # RNN layers
    for i in range(self.num_rnn):
      if self.is_bidirectional:
        layer = tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(units=self.rnn_units, dropout=0.2,
                               activation='tanh', recurrent_activation='sigmoid',
                               use_bias=True, recurrent_dropout=0.0,
                               return_sequences=True, unroll=False,
                               time_major=True, stateful=False, name=f"blstm_{i}"))(layer)
        layer = SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")(layer)
      else:
        layer = tf.keras.layers.LSTM(units=self.rnn_units, dropout=0.2,
                                     activation='tanh', recurrent_activation='sigmoid',
                                     use_bias=True, recurrent_dropout=0.0,
                                     return_sequences=True, unroll=False,
                                     time_major=False, stateful=streaming, name=f"lstm_{i}")(layer)
        layer = SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")(layer)
        if self.is_rowconv:
          layer = RowConv1D(filters=self.rnn_units, future_context=2, name=f"row_conv_{i}")(layer)

    # Convert to batch_major
    if self.is_bidirectional:
      layer = tf.transpose(layer, [1, 0, 2])

    if self.pre_fc_units > 0:
      layer = tf.keras.layers.Dense(units=self.pre_fc_units,
                                    name="pre_fully_connected",
                                    use_bias=True)(layer)
      layer = tf.keras.layers.BatchNormalization(name="pre_fc_bn")(layer)
      layer = tf.keras.layers.ReLU(name=f"relu_pre_fc")(layer)
      layer = tf.keras.layers.Dropout(0.2)(layer)

    return layer
