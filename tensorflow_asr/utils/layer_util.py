
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


def get_rnn(rnn_type: str):
    assert rnn_type in ["lstm", "gru", "rnn"]
    if rnn_type == "lstm": return tf.keras.layers.LSTM
    if rnn_type == "gru": return tf.keras.layers.GRU
    return tf.keras.layers.SimpleRNN


def get_conv(conv_type):
    assert conv_type in ["conv1d", "conv2d"]
    if conv_type == "conv1d": return tf.keras.layers.Conv1D
    return tf.keras.layers.Conv2D
