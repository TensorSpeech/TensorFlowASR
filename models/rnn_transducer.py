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


def create_transducer_model(base_model, num_classes, speech_conf,
                            last_activation='linear', streaming_size=None, name="transducer_model"):
    feature_dim, channel_dim = compute_feature_dim(speech_conf)
    if streaming_size:
        # Fixed input shape is required for live streaming audio
        x = tf.keras.layers.Input(batch_shape=(1, streaming_size, feature_dim, channel_dim),
                                  dtype=tf.float32, name="features")
        y = tf.keras.layers.Input(batch_shape=(1, None), dtype=tf.int32, name="predicted")
        # features = self.speech_featurizer(signal)
        outputs = base_model(x=x, y=y, streaming=True)
    else:
        x = tf.keras.layers.Input(shape=(None, feature_dim, channel_dim),
                                  dtype=tf.float32, name="features")
        y = tf.keras.layers.Input(batch_shape=(None,), dtype=tf.int32, name="predicted")
        # features = self.speech_featurizer(signal)
        outputs = base_model(x=x, y=y, streaming=False)

    # Fully connected layer
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=num_classes, activation=last_activation,
                              use_bias=True), name="fully_connected")(outputs)

    model = tf.keras.Model(inputs=[x, y], outputs=outputs, name=name)
    return model, base_model.optimizer
