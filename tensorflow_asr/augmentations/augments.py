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
import nlpaug.flow as naf

from .signal_augment import SignalCropping, SignalLoudness, SignalMask, SignalNoise, \
    SignalPitch, SignalShift, SignalSpeed, SignalVtlp
from .spec_augment import FreqMasking, TimeMasking, TFFreqMasking, TFTimeMasking


AUGMENTATIONS = {
    "freq_masking": FreqMasking,
    "time_masking": TimeMasking,
    "noise": SignalNoise,
    "masking": SignalMask,
    "cropping": SignalCropping,
    "loudness": SignalLoudness,
    "pitch": SignalPitch,
    "shift": SignalShift,
    "speed": SignalSpeed,
    "vtlp": SignalVtlp
}

TFAUGMENTATIONS = {
    "freq_masking": TFFreqMasking,
    "time_masking": TFTimeMasking,
}


class TFAugmentationExecutor:
    def __init__(self, augmentations: list, prob: float = 0.5):
        self.augmentations = augmentations
        self.prob = prob

    @tf.function
    def augment(self, inputs):
        outputs = inputs
        for au in self.augmentations:
            p = tf.random.uniform([])
            outputs = tf.where(tf.less(p, self.prob), au.augment(outputs), outputs)
        return outputs


class Augmentation:
    def __init__(self, config: dict = None, use_tf: bool = False):
        if not config: config = {}
        prob = float(config.pop("prob", 0.5))
        parser = self.tf_parse if use_tf else self.parse
        self.before = parser(config.pop("before", {}), prob=prob)
        self.after = parser(config.pop("after", {}), prob=prob)

    @staticmethod
    def parse(config: dict, prob: float = 0.5) -> naf.Sometimes:
        augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No augmentation named: {key}\n"
                               f"Available augmentations: {AUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return naf.Sometimes(augmentations, pipeline_p=prob)

    @staticmethod
    def tf_parse(config: dict, prob: float = 0.5) -> TFAugmentationExecutor:
        augmentations = []
        for key, value in config.items():
            au = TFAUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No tf augmentation named: {key}\n"
                               f"Available tf augmentations: {TFAUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return TFAugmentationExecutor(augmentations, prob=prob)
