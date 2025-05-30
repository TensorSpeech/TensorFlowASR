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

from typing import Optional

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_layer import Layer


@keras.utils.register_keras_serializable(package=__name__)
class Residual(Layer):
    """Applying residual addition to layers
    - Normal addition with constant factor
    - Rezero: which improves convergence speed. This implements the paper:
    ReZero is All You Need: Fast Convergence at Large Depth.
    (https://arxiv.org/pdf/2003.04887.pdf).
    """

    def __init__(
        self,
        factor="rezero",
        initializer: keras.initializers.Initializer = "zeros",
        regularizer: Optional[keras.regularizers.Regularizer] = None,
        name="residual",
        **kwargs,
    ):
        super().__init__(name=name, trainable=False, **kwargs)
        self._factor = factor
        self._initializer = initializer
        self._regularizer = regularizer

    def build(self, input_shape):
        if self._factor == "rezero":
            self._alpha = self.add_weight(
                name="alpha",
                shape=[],
                initializer=self._initializer,
                regularizer=self._regularizer,
                trainable=True,
                dtype=self.variable_dtype,
            )
        else:
            assert isinstance(self._factor, (int, float))
            self._alpha = self._factor
        return super().build(input_shape)

    def call(self, inputs):
        x, residual_x = inputs
        alpha = tf.cast(tf.convert_to_tensor(self._alpha, dtype=self.dtype), residual_x.dtype)
        x = x + alpha * residual_x
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]
