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

# Copyright 2021 Alexey Tochin
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
# ==============================================================================

import logging

from tensorflow_asr import tf
from tensorflow_asr.losses.base_loss import BaseLoss
from tensorflow_asr.losses.impl.ctc_tpu import classic_ctc_loss as ctc_loss_tpu

logger = logging.getLogger(__name__)


class CtcLoss(BaseLoss):
    def __init__(self, blank=0, reduction="sum_over_batch_size", name=None):
        super().__init__(blank=blank, reduction=reduction, name=name)
        logger.info("Use CTC loss TPU implementation" if self.use_tpu else "Use CTC loss")

    def call(self, y_true, y_pred):
        logits, logit_length, labels, label_length = super().call(y_true, y_pred)
        if self.use_tpu:
            return ctc_loss_tpu(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=logit_length,
                blank_index=self.blank,
            )
        return tf.nn.ctc_loss(
            logits=logits,
            logit_length=logit_length,
            labels=labels,
            label_length=label_length,
            logits_time_major=False,
            blank_index=self.blank,
            name=self.name,
        )
