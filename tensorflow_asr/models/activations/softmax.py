# pylint: disable=attribute-defined-outside-init
# Copyright 2024 Huy Le Nguyen (@nglehuy)
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


class Softmax(tf.keras.layers.Softmax):
    def call(self, inputs, mask=None):
        orig_dtype = inputs.dtype
        if orig_dtype == tf.float16:
            inputs = tf.cast(inputs, tf.float32)
        outputs = super().call(inputs, mask)
        if orig_dtype == tf.float16:
            outputs = tf.cast(outputs, orig_dtype)
        return outputs
