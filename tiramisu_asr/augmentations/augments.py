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

from collections import UserDict
import nlpaug.flow as naf

from .signal_augment import SignalCropping, SignalLoudness, SignalMask, SignalNoise, \
    SignalPitch, SignalShift, SignalSpeed, SignalVtlp
from .spec_augment import FreqMasking, TimeMasking


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


class UserAugmentation(UserDict):
    def __init__(self, config: dict = None):
        if not config: config = {}
        config["before"] = self.parse(config.get("before", {}))
        config["after"] = self.parse(config.get("after", {}))
        super(UserAugmentation, self).__init__(config)

    def __missing__(self, key):
        return None

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
