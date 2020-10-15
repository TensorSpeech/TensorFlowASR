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


class TimeReduction(tf.keras.layers.Layer):
    def __init__(self, factor: int, name: str = "time_reduction", **kwargs):
        super(TimeReduction, self).__init__(name=name, **kwargs)
        self.factor = factor

    def build(self, input_shape):
        batch_size = input_shape[0]
        feat_dim = input_shape[-1]
        self.reshape = tf.keras.layers.Reshape([batch_size, -1, feat_dim * self.factor])

    def call(self, inputs, **kwargs):
        return self.reshape(inputs)

    def get_config(self):
        config = super(TimeReduction, self).get_config()
        config.update({"factor": self.factor})
        return config

    def from_config(self, config):
        return self(**config)
