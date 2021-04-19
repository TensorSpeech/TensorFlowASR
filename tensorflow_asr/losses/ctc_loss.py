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


class CtcLoss(tf.keras.losses.Loss):
    def __init__(self, blank=0, global_batch_size=None, name=None):
        super(CtcLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.blank = blank
        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        loss = ctc_loss(
            y_pred=y_pred["logits"],
            input_length=y_pred["logits_length"],
            y_true=y_true["labels"],
            label_length=y_true["labels_length"],
            blank=self.blank,
            name=self.name
        )
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)


@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, blank, name=None):
    return tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits=tf.cast(y_pred, tf.float32),
        label_length=tf.cast(label_length, tf.int32),
        logits_time_major=False,
        blank_index=blank,
        name=name
    )
