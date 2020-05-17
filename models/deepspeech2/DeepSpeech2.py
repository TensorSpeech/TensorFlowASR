"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from models.deepspeech2.RowConv1D import RowConv1D
from models.deepspeech2.SequenceBatchNorm import SequenceBatchNorm

DEFAULT_CONV = {
  "conv_type": 2,
  "conv_kernels": ((11, 41), (11, 21), (11, 21)),
  "conv_strides": ((2, 2), (1, 2), (1, 2)),
  "conv_filters": (32, 32, 96),
  "conv_dropout": 0.2
}

DEFAULT_RNN = {
  "rnn_layers": 3,
  "rnn_type": "gru",
  "rnn_units": 350,
  "rnn_bidirectional": True,
  "rnn_rowconv": False,
  "rnn_rowconv_context": 2,
  "rnn_dropout": 0.2
}

DEFAULT_FC = {
  "fc_units": (1024,),
  "fc_dropout": 0.2
}


class DeepSpeech2:
  def __init__(self, conv_conf=DEFAULT_CONV, rnn_conf=DEFAULT_RNN, fc_conf=DEFAULT_FC,
               optimizer=tf.keras.optimizers.SGD(lr=0.0002, momentum=0.99, nesterov=True)):
    self.optimizer = optimizer
    self.conv_conf = conv_conf
    self.rnn_conf = rnn_conf
    self.fc_conf = fc_conf
    assert len(conv_conf["conv_strides"]) == len(conv_conf["conv_filters"]) == len(conv_conf["conv_kernels"])
    assert conv_conf["conv_type"] in [1, 2]
    assert rnn_conf["rnn_type"] in ["lstm", "gru", "rnn"]
    assert conv_conf["conv_dropout"] >= 0.0 and rnn_conf["rnn_dropout"] >= 0.0

  @staticmethod
  def merge_filter_to_channel(x):
    batch_size = tf.shape(x)[0]
    f, c = x.get_shape().as_list()[2:]
    return tf.reshape(x, [batch_size, -1, f * c])

  @staticmethod
  def merge_batch_and_time(x):
    batch_size = tf.shape(x)[0]
    feat = x.get_shape().as_list()[-1]
    return tf.reshape(x, [-1, feat]), batch_size

  @staticmethod
  def undo_merge_batch_and_time(x, batch_size):
    feat = x.get_shape().as_list()[-1]
    return tf.reshape(x, [batch_size, -1, feat])

  def __call__(self, features, streaming=False):
    layer = features
    if self.conv_conf["conv_type"] == 2:
      conv = tf.keras.layers.Conv2D
    else:
      layer = self.merge_filter_to_channel(layer)
      conv = tf.keras.layers.Conv1D
      ker_shape = np.shape(self.conv_conf["conv_kernels"])
      stride_shape = np.shape(self.conv_conf["conv_strides"])
      filter_shape = np.shape(self.conv_conf["conv_filters"])
      assert len(ker_shape) == 1 and len(stride_shape) == 1 and len(filter_shape) == 1

    for i, fil in enumerate(self.conv_conf["conv_filters"]):
      layer = conv(filters=fil, kernel_size=self.conv_conf["conv_kernels"][i],
                   strides=self.conv_conf["conv_strides"][i], padding="same",
                   activation=None, name=f"cnn_{i}")(layer)
      layer = tf.keras.layers.BatchNormalization(name=f"cnn_bn_{i}")(layer)
      layer = tf.keras.layers.ReLU(name=f"cnn_relu_{i}")(layer)
      layer = tf.keras.layers.Dropout(self.conv_conf["conv_dropout"], name=f"cnn_dropout_{i}")(layer)

    if self.conv_conf["conv_type"] == 2:
      layer = self.merge_filter_to_channel(layer)

    # Convert to time_major only for bi_directional
    if self.rnn_conf["rnn_bidirectional"]:
      layer = tf.transpose(layer, [1, 0, 2])

    if self.rnn_conf["rnn_type"] == "rnn":
      rnn_cell = tf.keras.layers.SimpleRNNCell(self.rnn_conf["rnn_units"], activation="tanh",
                                               use_bias=True, dropout=self.rnn_conf["rnn_dropout"])
    elif self.rnn_conf["rnn_type"] == "lstm":
      rnn_cell = tf.keras.layers.LSTMCell(self.rnn_conf["rnn_units"], activation="tanh", use_bias=True,
                                          recurrent_activation="sigmoid", dropout=self.rnn_conf["rnn_dropout"])
    else:
      rnn_cell = tf.keras.layers.GRUCell(self.rnn_conf["rnn_units"], activation="tanh", use_bias=True,
                                         recurrent_activation="sigmoid", dropout=self.rnn_conf["rnn_dropout"])

    # RNN layers
    for i in range(self.rnn_conf["rnn_layers"]):
      if self.rnn_conf["rnn_bidirectional"]:
        layer = tf.keras.layers.Bidirectional(
          tf.keras.layers.RNN(rnn_cell, return_sequences=True, unroll=False,
                              time_major=True, stateful=False), name=f"b{self.rnn_conf['rnn_type']}_{i}")(layer)
        layer = SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")(layer)
      else:
        tf.keras.layers.RNN(rnn_cell, return_sequences=True, unroll=False,
                            time_major=False, stateful=False, name=f"{self.rnn_conf['rnn_type']}_{i}")(layer)
        layer = SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")(layer)
        if self.rnn_conf["rnn_rowconv"]:
          layer = RowConv1D(filters=self.rnn_conf["rnn_units"],
                            future_context=self.rnn_conf["rnn_rowconv_context"], name=f"row_conv_{i}")(layer)

    # Convert to batch_major
    if self.rnn_conf["rnn_bidirectional"]:
      layer = tf.transpose(layer, [1, 0, 2])

    if self.fc_conf["fc_units"]:
      assert self.fc_conf["fc_dropout"] >= 0.0

      layer, batch_size = self.merge_batch_and_time(layer)

      for idx, units in enumerate(self.fc_conf["fc_units"]):
        layer = tf.keras.layers.Dense(units=units, activation=None,
                                      use_bias=True, name=f"hidden_fc_{idx}")(layer)
        layer = tf.keras.layers.BatchNormalization(name=f"hidden_fc_bn_{idx}")(layer)
        layer = tf.keras.layers.ReLU(name=f"hidden_fc_relu_{idx}")(layer)
        layer = tf.keras.layers.Dropout(self.fc_conf["fc_dropout"], name=f"hidden_fc_dropout_{idx}")(layer)

      layer = self.undo_merge_batch_and_time(layer, batch_size)

    return layer
