from __future__ import absolute_import

import tensorflow as tf


class Custom:
  def __init__(self, num_rnn=2, rnn_units=512, denses=(64, 128, 256),
               filters=(16, 32, 64, 128, 256), kernel_size=31,
               optimizer=tf.keras.optimizers.SGD(lr=0.0002, momentum=0.99, nesterov=True)):
    self.optimizer = optimizer
    self.num_rnn = num_rnn
    self.rnn_units = rnn_units
    self.filters = filters
    self.kernel_size = kernel_size
    self.denses = denses
    if filters[-1] != denses[-1]:
      raise ValueError("Last filter must equal to last dense")

  def __call__(self, features, streaming=False):
    h = features
    x = h
    for i, fil in enumerate(self.filters):
      h = tf.keras.layers.Conv1D(filters=fil, kernel_size=self.kernel_size,
                                 strides=2, padding="same", name=f"cnn_{i}")(h)
      h = tf.keras.layers.ReLU(max_value=20, name=f"relu_cnn_{i}")(h)

    for i, den in enumerate(self.denses):
      x = tf.keras.layers.Dense(units=den, use_bias=True, name=f"dense_{i}")(x)
      x = tf.keras.layers.ReLU(max_value=20, name=f"relu_dense_{i}")(x)

    x = tf.keras.layers.Attention(name="attention")([x, h])

    # RNN layers
    for i in range(self.num_rnn):
      x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=self.rnn_units, dropout=0.2,
                             activation='tanh', recurrent_activation='sigmoid',
                             use_bias=True, recurrent_dropout=0.0,
                             return_sequences=True, unroll=False, implementation=2,
                             time_major=True, stateful=False, name=f"blstm_{i}"))(x)

    return x
