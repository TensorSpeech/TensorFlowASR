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
                tf.zeros_like(g),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
            ) for g in trainable_variables
        ]

    def reset(self):
        for i, g in enumerate(self.gradients):
            self.gradients[i].assign(tf.zeros_like(g), read_value=False)

    def accumulate(self, step_gradients):
        for i, g in enumerate(step_gradients):
            if g is None: continue
            self.gradients[i].assign_add(g, read_value=False)
