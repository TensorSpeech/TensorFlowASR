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


@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, num_classes):
    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logit_length=input_length,
        logits=tf.cast(y_pred, tf.float32),
        label_length=label_length,
        logits_time_major=False,
        blank_index=num_classes - 1
    )
    return tf.reduce_mean(loss)

# @tf.function
# def ctc_loss_keras(layer, **kwargs):
#     num_classes = kwargs["num_classes"]
#     y_pred, input_length, y_true, label_length = layer
#     loss = tf.nn.ctc_loss(
#         labels=y_true,
#         logit_length=input_length,
#         logits=tf.cast(y_pred, tf.float32),
#         label_length=label_length,
#         logits_time_major=False,
#         blank_index=num_classes - 1
#     )
#     return tf.reduce_mean(loss)
