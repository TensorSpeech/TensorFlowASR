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

from tensorflow_asr import keras
from tensorflow_asr.utils import math_util


@keras.utils.register_keras_serializable(package=__name__)
class Layer(keras.layers.Layer):
    def __init__(
        self,
        trainable=True,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.utils.register_keras_serializable(package=__name__)
class Reshape(Layer):
    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = output_shape[:2] + (output_shape[2] * output_shape[3],)
        return output_shape, output_length_shape
