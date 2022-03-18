# Copyright 2022 Huy Le Nguyen (@usimarit)
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

from tensorflow_asr.models.layers.base_layer import Layer


class OneHotBlank(Layer):
    """
    https://arxiv.org/pdf/1211.3711.pdf
    The inputs are encoded as one-hot vectors;
    that is, if Y consists of K labels and yu = k, then y^u is a length K vector whose elements are all zero
    except the k-th, which is one. ∅ is encoded as a length K vector of zeros
    """

    def __init__(self, blank, depth, name="one_hot_blank", **kwargs):
        super().__init__(name=name, **kwargs)
        self.blank = blank
        self.depth = depth

    def call(self, inputs, training=False):
        minus_one_at_blank = tf.where(tf.equal(inputs, self.blank), -1, inputs)
        return tf.one_hot(minus_one_at_blank, depth=self.depth, dtype=self.dtype)
