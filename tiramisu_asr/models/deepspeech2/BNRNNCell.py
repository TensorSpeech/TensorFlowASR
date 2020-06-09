# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops


def ds2_rnn_batch_norm(x_i, x_f, x_c, x_o, beta=None, gamma=None):
    # x is input * weight with shape [batch_size, units * 4]
    # Merge into single array of features
    # https://www.tensorflow.org/api_docs/python/tf/nn/moments
    x = tf.concat([x_i, x_f, x_c, x_o], axis=1)
    mean, variance = tf.nn.moments(x, axes=[0, 1], keepdims=False)
    x = tf.nn.batch_normalization(x=x, mean=mean, variance=variance,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=K.epsilon())
    x_i, x_f, x_c, x_o = array_ops.split(x, num_or_size_splits=4,
                                         axis=1)
    return x_i, x_f, x_c, x_o


# Frame-wise Batch Norm RNN
class BNLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = self.add_weight(shape=(self.units * 4,),
                                    name='lstm_bn_beta', initializer='zeros',
                                    regularizer=None, constraint=None, trainable=True)
        self.gamma = self.add_weight(shape=(self.units * 4,),
                                     name='lstm_bn_gamma', initializer='ones',
                                     regularizer=None, constraint=None, trainable=True)

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        x_i, x_f, x_c, x_o = ds2_rnn_batch_norm(x_i, x_f, x_c, x_o,
                                                beta=self.beta,
                                                gamma=self.gamma)

        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c,
            self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o
