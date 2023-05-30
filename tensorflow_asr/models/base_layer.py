# Copyright 2022 Huy Le Nguyen (@nglehuy)
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

import keras.layers
from keras.utils import tf_utils

from tensorflow_asr.utils import math_util


class Layer(keras.layers.Layer):
    def __init__(
        self,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self._output_shape = None
        self.supports_masking = True

    @property
    def output_shape(self):
        if self._output_shape is None:
            raise AttributeError(f"The layer {self.name} has never been called and thus has no defined output shape.")
        return self._output_shape

    def build(self, input_shape):
        self._output_shape = tf_utils.convert_shapes(self.compute_output_shape(input_shape), to_tuples=True)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class Reshape(Layer):
    def call(self, inputs):
        return math_util.merge_two_last_dims(inputs)

    def compute_output_shape(self, input_shape):
        b, h, w, d = input_shape
        return (b, h, w * d)
