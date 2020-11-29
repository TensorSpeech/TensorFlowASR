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


class PointWiseFFN(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 output_size,
                 activation="relu",
                 dropout=0.1,
                 name="point_wise_ffn",
                 **kwargs):
        super(PointWiseFFN, self).__init__(name=name, **kwargs)
        self.ffn1 = tf.keras.layers.Dense(units=size, activation=activation)
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(units=output_size)
        self.do2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ffn1(inputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(PointWiseFFN, self).get_config()
        conf.update(self.ffn1.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        return conf
