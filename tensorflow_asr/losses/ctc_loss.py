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

from tensorflow_asr import keras, tf
from tensorflow_asr.losses.base_loss import BaseLoss

logger = tf.get_logger()


class CtcLoss(BaseLoss):
    def __init__(self, blank=0, reduction=keras.losses.Reduction.AUTO, name=None):
        super().__init__(blank=blank, reduction=reduction, name=name)
        logger.info("Use CTC loss")

    def call(self, y_true, y_pred):
        logits, logit_length, labels, label_length = super().call(y_true, y_pred)
        labels = labels if self.use_tpu else tf.sparse.from_dense(labels)
        unique = tf.nn.ctc_unique_labels(labels) if self.use_tpu else None
        return tf.nn.ctc_loss(
            logits=logits,
            logit_length=logit_length,
            labels=labels,
            label_length=label_length,
            logits_time_major=False,
            unique=unique,  # enable a faster, memory efficient implementation on TPU.
            blank_index=self.blank,
            name=self.name,
        )
