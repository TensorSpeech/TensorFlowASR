from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops


def ds2_rnn_batch_norm(x_i, x_f, x_c, x_o, beta=None, gamma=None):
    # x is input * weight with shape [batch_size, features]
    # Merge into single array of features
    # https://www.tensorflow.org/api_docs/python/tf/nn/moments
    x = tf.concat([x_i, x_f, x_c, x_o], axis=1)
    mean, variance = tf.nn.moments(x, axis=[0], keepdims=False)
    x = tf.nn.batch_normalization(x=x, mean=mean, variance=variance,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=K.epsilon())
    x_i, x_f, x_c, x_o = array_ops.split(x, num_or_size_splits=4, axis=1)
    return (x_i, x_f, x_c, x_o)


class BNLSTMCell(tf.keras.layers.LSTMCell):
    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        x_i, x_f, x_c, x_o = ds2_rnn_batch_norm(x_i, x_f, x_c, x_o)

        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o
