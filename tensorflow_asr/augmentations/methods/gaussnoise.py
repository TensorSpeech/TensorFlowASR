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

import tensorflow as tf

from tensorflow_asr.augmentations.methods.base_method import AugmentationMethod


class GaussNoise(AugmentationMethod):
    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.075,
        prob: float = 0.5,
    ):
        super().__init__(prob=prob)
        self.mean = mean
        self.stddev = stddev

    def augment(self, args):
        inputs, inputs_length = args
        prob = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        do_apply = tf.where(tf.less_equal(prob, self.prob), tf.constant(1, inputs.dtype), tf.constant(0, inputs.dtype))
        noise = tf.random.normal(shape=tf.shape(inputs), mean=self.mean, stddev=self.stddev, dtype=inputs.dtype)
        noise *= tf.sequence_mask(inputs_length, inputs.shape[1], dtype=inputs.dtype)
        noise *= do_apply
        return tf.add(inputs, noise), inputs_length
