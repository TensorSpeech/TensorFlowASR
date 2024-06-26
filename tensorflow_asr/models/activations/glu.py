# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_layer import Layer


@keras.utils.register_keras_serializable(package=__name__)
class GLU(Layer):
    def __init__(self, axis=-1, name="glu", **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def compute_output_shape(self, input_shape):
        B, T, V = input_shape
        return (B, T, V // 2)
