# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr.utils import math_util


class Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.supports_masking = True


class Reshape(Layer):
    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = output_shape[:2] + (output_shape[2] * output_shape[3],)
        return output_shape, output_length_shape


class Identity(Layer):
    def call(self, inputs):
        return inputs
