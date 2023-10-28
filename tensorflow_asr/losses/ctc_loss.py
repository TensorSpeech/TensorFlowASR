# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

logger = tf.get_logger()


class CtcLoss(tf.keras.losses.Loss):
    def __init__(self, blank=0, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.blank = blank
        logger.info("Use CTC loss")

    def call(self, y_true, y_pred):
        return tf.nn.ctc_loss(
            logits=y_pred,
            logit_length=y_pred._keras_length,
            labels=y_true,
            label_length=y_true._keras_length,
            logits_time_major=False,
            unique=tf.nn.ctc_unique_labels(y_true),  # enable a faster, memory efficient implementation on TPU.
            blank_index=self.blank,
            name=self.name,
        )
