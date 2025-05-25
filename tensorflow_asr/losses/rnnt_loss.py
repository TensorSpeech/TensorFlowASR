# pylint: disable=no-name-in-module,unexpected-keyword-arg,no-value-for-parameter
# Copyright 2020 Huy Le Nguyen (@nglehuy) and M. Yusuf Sarıgöz (@monatis)
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
# RNNT loss implementation in pure TensorFlow is borrowed from [iamjanvijay's repo](https://github.com/iamjanvijay/rnnt)


import logging
import os

from tensorflow_asr.losses.base_loss import BaseLoss
from tensorflow_asr.losses.impl.rnnt import rnnt_loss, warp_rnnt_loss
from tensorflow_asr.utils import env_util

TFASR_USE_CPU_LOSS = os.getenv("TFASR_USE_CPU_LOSS", "False") in ("true", "True", "1")

logger = logging.getLogger(__name__)


class RnntLoss(BaseLoss):
    def __init__(
        self,
        blank,
        reduction="sum_over_batch_size",
        output_shapes=None,
        name=None,
    ):
        super().__init__(blank=blank, reduction=reduction, name=name)
        self.use_cpu = TFASR_USE_CPU_LOSS or (not env_util.has_devices("GPU") and not env_util.has_devices("TPU"))
        self.output_shapes = output_shapes
        # fmt: off
        logger.info(f"[RNNT loss] Use {'CPU' if self.use_cpu else 'GPU/TPU'} implementation in {'Tensorflow' if warp_rnnt_loss is None else 'WarpRNNT'}") # pylint: disable=line-too-long
        # fmt: on
        if self.output_shapes:
            logger.info(f"[RNNT loss] Use model's output shapes: {self.output_shapes}")
            if not all(self.output_shapes):
                logger.info("[RNNT loss] Detected dynamic shape")
                self.output_shapes = None

    def call(self, y_true, y_pred):
        logits, logit_length, labels, label_length = super().call(y_true, y_pred)
        return rnnt_loss(
            logits=logits,
            logits_length=logit_length,
            labels=labels,
            labels_length=label_length,
            blank=self.blank,
            name=self.name,
            use_cpu=self.use_cpu,
            output_shapes=self.output_shapes,
        )

    def get_config(self):
        conf = super().get_config()
        conf.update({"output_shapes": self.output_shapes})
        return conf
