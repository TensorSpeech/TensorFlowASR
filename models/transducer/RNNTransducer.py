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
from __future__ import absolute_import

import tensorflow as tf

DEFAULT_ENCODER = {
    "cnn_type": 2,
    "cnn_kernel": ((11, 41), (11, 21), (11, 21)),
    "cnn_strides": ((2, 2), (1, 2), (1, 2)),
    "cnn_filters": (32, 32, 96),
    "rnn_type": "lstm",
    "rnn_units": 256,
    "rnn_layers": 3,
    "rnn_bidirectional": True
}

DEFAULT_PREDICTION_NET = {
    "embedding_size": 256,
    "rnn_type": "lstm",
    "rnn_units": 256,
    "rnn_layers": 3,
    "rnn_bidirectional": True
}


class RNNTransducer:
    def __init__(self, encoder_conf=DEFAULT_ENCODER, prediction_conf=DEFAULT_PREDICTION_NET, joint_units=1024,
                 dropout=0.2, optimizer=tf.keras.optimizers.SGD(lr=0.0002, momentum=0.99, nesterov=True)):
        self.optimizer = optimizer
        self.encoder_conf = encoder_conf
        self.prediction_conf = prediction_conf
        self.dropout = dropout
        self.joint_units = joint_units

    def get_rnn_cell(self, rnn_type):
        assert rnn_type in ["lstm", "gru", "rnn"]
        if rnn_type == "lstm":
            rnn_cell = tf.keras.layers.LSTMCell(self.encoder_conf["rnn_units"], dropout=self.dropout)
        elif rnn_type == "gru":
            rnn_cell = tf.keras.layers.GRUCell(self.encoder_conf["rnn_units"], dropout=self.dropout)
        else:
            rnn_cell = tf.keras.layers.SimpleRNNCell(self.encoder_conf["rnn_units"], dropout=self.dropout)
        return rnn_cell

    @staticmethod
    def merge_features_to_channels(x):
        batch_size = tf.shape(x)[0]
        f, c = x.get_shape().as_list()[2:]
        return tf.reshape(x, [batch_size, -1, f * c])

    def encoder_network(self, x):
        assert self.encoder_conf["cnn_type"] in [1, 2]
        if self.encoder_conf["cnn_type"] == 1:
            conv = tf.keras.layers.Conv1D
            x = self.merge_features_to_channels(x)
        else:
            conv = tf.keras.layers.Conv2D

        for idx, kernel in enumerate(self.encoder_conf["cnn_kernel"]):
            x = conv(filters=self.encoder_conf["cnn_filters"][idx], kernel_size=kernel,
                     strides=self.encoder_conf["cnn_strides"][idx], padding="same",
                     name=f"encoder_cnn_{idx}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"encoder_cnn_bn_{idx}")(x)
            x = tf.keras.layers.ReLU(name=f"encoder_cnn_relu_{idx}")(x)
            x = tf.keras.layers.Dropout(self.dropout, name=f"encoder_cnn_dropout_{idx}")(x)

        if self.encoder_conf["cnn_type"] == 2:
            x = self.merge_features_to_channels(x)

        rnn_cell = self.get_rnn_cell(self.encoder_conf["rnn_type"])

        for idx in range(self.encoder_conf["rnn_layers"]):
            if self.encoder_conf["rnn_bidirectional"]:
                x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.RNN(rnn_cell, return_sequences=True, stateful=False, time_major=False),
                    name=f"b{self.encoder_conf['rnn_type']}_{idx}"
                )(x)
            else:
                x = tf.keras.layers.RNN(rnn_cell, return_sequences=True,
                                        stateful=False, time_major=False,
                                        name=f"{self.encoder_conf['rnn_type']}_{idx}")(x)
            x = tf.keras.layers.LayerNormalization(name=f"{self.encoder_conf['rnn_type']}_ln_{idx}")(x)

        return x

    def prediction_network(self, y, num_classes):
        y = tf.keras.layers.Embedding(num_classes - 1, self.prediction_conf["embedding_size"],
                                      name="prediction_embedding")(y)

        rnn_cell = self.get_rnn_cell(self.prediction_conf["rnn_type"])

        for idx in range(self.prediction_conf["rnn_layers"]):
            if self.prediction_conf["rnn_bidirectional"]:
                y = tf.keras.layers.Bidirectional(
                    tf.keras.layers.RNN(rnn_cell, return_sequences=True, stateful=False, time_major=False),
                    name=f"b{self.prediction_conf['rnn_type']}_{idx}"
                )(y)
            else:
                y = tf.keras.layers.RNN(rnn_cell, return_sequences=True,
                                        stateful=False, time_major=False,
                                        name=f"{self.prediction_conf['rnn_type']}_{idx}")(y)
            y = tf.keras.layers.LayerNormalization(name=f"{self.prediction_conf['rnn_type']}_ln_{idx}")(y)

        return y

    def __call__(self, x, y, num_classes, streaming=False):
        x = self.encoder_network(x)
        y = self.prediction_network(y, num_classes)

        x_exp = tf.expand_dims(x, axis=2)  # [B, T, V] => [B, T, 1, V]
        y_exp = tf.expand_dims(y, axis=1)  # [B, U, V] => [B, 1, U, V]

        x = tf.tile(x_exp, tf.stack([1, 1, tf.shape(x)[1], 1], name="stack_x"))
        y = tf.tile(y_exp, tf.stack([1, tf.shape(y)[1], 1, 1], name="stack_y"))

        joint = tf.keras.layers.Concatenate(axis=-1)([x, y])

        return tf.keras.layers.Dense(self.joint_units, activation="tanh", name="joint_fnn")(joint)
