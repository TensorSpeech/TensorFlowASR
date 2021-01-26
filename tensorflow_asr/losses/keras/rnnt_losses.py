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
from tensorflow.python.keras.utils import losses_utils

from .. import rnnt_loss


class RnntLoss(tf.keras.losses.Loss):
    def __init__(self, blank=0, global_batch_size=None, reduction=losses_utils.ReductionV2.NONE, name=None):
        super(RnntLoss, self).__init__(reduction=reduction, name=name)
        self.blank = blank
        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        logits = y_pred["logit"]
        logit_length = y_pred["logit_length"]
        labels = y_true["label"]
        label_length = y_true["label_length"]
        loss = rnnt_loss(logits, labels, label_length, logit_length, blank=self.blank)
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)
