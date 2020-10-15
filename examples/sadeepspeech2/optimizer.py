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

from tensorflow_asr.optimizers.schedules import TransformerSchedule, SANSchedule


def create_optimizer(name, d_model, lamb=0.05, warmup_steps=4000):
    if name == "transformer_adam":
        learning_rate = TransformerSchedule(d_model, warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                             beta_2=0.98, epsilon=1e-9)

    elif name == "transformer_sgd":
        learning_rate = TransformerSchedule(d_model, warmup_steps)
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.99, nesterov=True)

    elif name == "san":
        learning_rate = SANSchedule(lamb, d_model, warmup_steps)
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.99, nesterov=True)

    else:
        raise ValueError("optimizer name must be either 'transformer' or 'san'")

    return optimizer
