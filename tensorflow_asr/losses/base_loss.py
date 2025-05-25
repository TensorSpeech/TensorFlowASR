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

from tensorflow_asr import keras, schemas, tf
from tensorflow_asr.utils import env_util

logger = tf.get_logger()


class BaseLoss(keras.losses.Loss):
    def __init__(self, blank=0, reduction="sum_over_batch_size", name=None):
        super().__init__(reduction=reduction, name=name)
        assert blank == 0, "Only support blank=0"
        self.blank = blank
        self.use_tpu = env_util.has_devices("TPU")

    def call(
        self,
        y_true: schemas.TrainLabel,
        y_pred: schemas.TrainOutput,
    ):
        logit_length = tf.cast(y_pred.logits_length, tf.int32)
        labels = tf.cast(y_true.labels, tf.int32)
        label_length = tf.cast(y_true.labels_length, tf.int32)
        logit_length = tf.where(tf.less(logit_length, label_length), label_length, logit_length)  # pad logit_length to label_length
        return y_pred.logits, logit_length, labels, label_length

    def get_config(self):
        config = super().get_config()
        config.update({"blank": self.blank})
        return config
