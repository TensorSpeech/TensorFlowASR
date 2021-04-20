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

from .methods import specaugment


AUGMENTATIONS = {
    "freq_masking": specaugment.FreqMasking,
    "time_masking": specaugment.TimeMasking,
}


class Augmentation:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.prob = float(config.pop("prob", 0.5))
        self.signal_augmentations = self.parse(config.pop("signal_augment", {}))
        self.feature_augmentations = self.parse(config.pop("feature_augment", {}))

    def _augment(self, inputs, augmentations):
        outputs = inputs
        for au in augmentations:
            p = tf.random.uniform([])
            outputs = tf.where(tf.less(p, self.prob), au.augment(outputs), outputs)
        return outputs

    @tf.function
    def signal_augment(self, inputs):
        return self._augment(inputs, self.signal_augmentations)

    @tf.function
    def feature_augment(self, inputs):
        return self._augment(inputs, self.feature_augmentations)

    @staticmethod
    def parse(config: dict) -> list:
        augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No tf augmentation named: {key}\n"
                               f"Available tf augmentations: {AUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return augmentations
