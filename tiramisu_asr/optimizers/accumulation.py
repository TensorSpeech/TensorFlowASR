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


class GradientAccumulation:
    def __init__(self, trainable_variables):
        self.gradients = [
            tf.Variable(
                tf.zeros_like(self.flat_gradients(g)),
                synchronization=tf.VariableSynchronization.ON_READ
            ) for g in trainable_variables
        ]

    @staticmethod
    def flat_gradients(gradient):
        """ Convert gradients if it's tf.IndexedSlices. """
        if type(gradient) == tf.IndexedSlices:
            return tf.scatter_nd(
                tf.expand_dims(gradient.indices, 1),
                gradient.values,
                gradient.dense_shape
            )
        return gradient

    def reset(self):
        for g in self.gradients: g.assign(tf.zeros_like(g))

    def accumulate(self, step_gradients):
        for i, g in enumerate(step_gradients):
            self.gradients[i].assign_add(self.flat_gradients(g))
