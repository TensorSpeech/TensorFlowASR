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
    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    @tf.function
    def augment(self, inputs):
        outputs = inputs
        for au in self.augmentations:
            outputs = au.augment(outputs)
        return outputs


class Augmentation:
    def __init__(self, config: dict = None, use_tf: bool = False):
        if not config: config = {}
        if use_tf:
            self.before = self.tf_parse(config.pop("before", {}))
            self.after = self.tf_parse(config.pop("after", {}))
        else:
            self.before = self.parse(config.pop("before", {}))
            self.after = self.parse(config.pop("after", {}))

    @staticmethod
    def parse(config: dict) -> list:
        augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No augmentation named: {key}\n"
                               f"Available augmentations: {AUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return naf.Sometimes(augmentations)

    @staticmethod
    def tf_parse(config: dict) -> list:
        augmentations = []
        for key, value in config.items():
            au = TFAUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No tf augmentation named: {key}\n"
                               f"Available tf augmentations: {TFAUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return TFAugmentationExecutor(augmentations)
