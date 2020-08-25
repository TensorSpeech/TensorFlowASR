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
from warprnnt_tensorflow import rnnt_loss as warp_rnnt_loss


@tf.function
def rnnt_loss(logits, labels, label_length, logit_length, blank=0):
    if not tf.config.list_physical_devices('GPU'):
        logits = tf.nn.log_softmax(logits)
    loss = warp_rnnt_loss(
        acts=tf.cast(logits, tf.float32),
        label_lengths=tf.cast(label_length, tf.int32),
        labels=tf.cast(labels, tf.int32),
        input_lengths=tf.cast(logit_length, tf.int32),
        blank_label=blank
    )
    return loss
