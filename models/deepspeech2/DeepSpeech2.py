"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
from __future__ import absolute_import

import tensorflow as tf
from models.deepspeech2.RowConv1D import RowConv1D
from models.deepspeech2.SequenceBatchNorm import SequenceBatchNorm


class DeepSpeech2:
  def __init__(self, num_rnn=5, rnn_units=256, filters=(32, 32, 96),
               kernel_size=((11, 41), (11, 21), (11, 21)), strides=((2, 2), (1, 2), (1, 2)),
               optimizer=tf.keras.optimizers.SGD(lr=0.0002, momentum=0.99, nesterov=True),
               is_bidirectional=False, is_rowconv=False, pre_fc_units=1024):
    self.optimizer = optimizer
    self.num_rnn = num_rnn
    self.rnn_units = rnn_units
    self.filters = filters
    self.kernel_size = kernel_size
    self.is_bidirectional = is_bidirectional
    self.is_rowconv = is_rowconv
    self.pre_fc_units = pre_fc_units
    self.strides = strides
    if len(strides) != len(filters) != len(kernel_size):
      raise ValueError("Strides must equal Filters")

  def __call__(self, features, streaming=False):
    layer = tf.expand_dims(features, -1)
    for i, fil in enumerate(self.filters):
      layer = tf.keras.layers.Conv2D(filters=fil, kernel_size=self.kernel_size[i],
                                     strides=self.strides[i], padding="same", name=f"cnn_{i}")(layer)
      layer = tf.keras.layers.BatchNormalization(name=f"bn_cnn_{i}")(layer)
      layer = tf.keras.layers.ReLU(name=f"relu_cnn_{i}")(layer)

    batch_size = tf.shape(layer)[0]
    f, c = layer.get_shape().as_list()[2:]
    layer = tf.reshape(layer, [batch_size, -1, f * c])

    # Convert to time_major only for bi_directional
    if self.is_bidirectional:
      layer = tf.transpose(layer, [1, 0, 2])

    # RNN layers
    for i in range(self.num_rnn):
      if self.is_bidirectional:
        layer = tf.keras.layers.Bidirectional(
          tf.keras.layers.GRU(units=self.rnn_units, dropout=0.2,
                              activation='tanh', recurrent_activation='sigmoid',
                              use_bias=True, recurrent_dropout=0.0,
                              return_sequences=True, unroll=False, implementation=2,
                              time_major=True, stateful=False, name=f"blstm_{i}"))(layer)
        layer = SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")(layer)
      else:
        layer = tf.keras.layers.GRU(units=self.rnn_units, dropout=0.2,
                                    activation='tanh', recurrent_activation='sigmoid',
                                    use_bias=True, recurrent_dropout=0.0,
                                    return_sequences=True, unroll=False, implementation=2,
                                    time_major=False, stateful=streaming, name=f"lstm_{i}")(layer)
        layer = SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")(layer)
        if self.is_rowconv:
          layer = RowConv1D(filters=self.rnn_units, future_context=2, name=f"row_conv_{i}")(layer)

    # Convert to batch_major
    if self.is_bidirectional:
      layer = tf.transpose(layer, [1, 0, 2])

    if self.pre_fc_units > 0:
      layer = tf.keras.layers.Dense(units=self.pre_fc_units,
                                    name="hidden_fc",
                                    use_bias=True)(layer)
      layer = tf.keras.layers.BatchNormalization(name="hidden_fc_bn")(layer)
      layer = tf.keras.layers.ReLU(name="hidden_fc_relu")(layer)
      layer = tf.keras.layers.Dropout(0.2, name="hidden_fc_dropout")(layer)

    return layer
