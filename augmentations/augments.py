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
from __future__ import absolute_import

import os
import abc
import glob
import librosa
import random
from collections import UserDict
from augmentations.spec_augment import time_masking, freq_masking
from augmentations.noise_augment import add_noise, add_white_noise, add_realworld_noise


class Augmentation(metaclass=abc.ABCMeta):
    def __init__(self, params: dict):
        self.params = params

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.params)

    @abc.abstractmethod
    def func(self, *args, **kwargs):
        """ Function to perform augmentation """
        pass


class FreqMasking(Augmentation):
    def func(self, *args, **kwargs):
        return freq_masking(*args, **kwargs)


class TimeMasking(Augmentation):
    def func(self, *args, **kwargs):
        return time_masking(*args, **kwargs)


# class TimeWarping(Augmentation):
#     def __init__(self, time_warp_param: int = 50):
#         self.time_warp_param = time_warp_param
#         super(TimeWarping, self).__init__(
#             func=functools.partial(time_warping, time_warp_param=self.time_warp_param),
#             is_post=True
#         )


class Noise(Augmentation):
    def __init__(self, params: dict):
        params["snr_list"] = list(params["snr_list"])
        if not any([i < 0 for i in params["snr_list"]]):
            params["snr_list"].append(-1)
        params["noise_dir"] = glob.glob(os.path.join(params["noise_dir"], "**", "*.wav"), recursive=True)
        super(Noise, self).__init__(params)

    def func(self, *args, **kwargs):
        return add_noise(*args, **kwargs)


class WhiteNoise(Augmentation):
    def __init__(self, params: dict):
        params["snr_list"] = list(params["snr_list"])
        if not any([i < 0 for i in params["snr_list"]]):
            params["snr_list"].append(-1)
        super(WhiteNoise, self).__init__(params)

    def func(self, *args, **kwargs):
        return add_white_noise(*args, **kwargs)


class RealWorldNoise(Augmentation):
    def __init__(self, params: dict):
        params["snr_list"] = list(params["snr_list"])
        if not any([i < 0 for i in params["snr_list"]]):
            params["snr_list"].append(-1)
        params["noise_dir"] = glob.glob(os.path.join(params["noise_dir"], "**", "*.wav"), recursive=True)
        super(RealWorldNoise, self).__init__(params)

    def func(self, *args, **kwargs):
        return add_realworld_noise(*args, **kwargs)


class TimeStretch(Augmentation):
    def func(self, *args, **kwargs):
        rate = random.uniform(kwargs["min_ratio"], kwargs["max_ratio"])
        return librosa.effects.time_stretch(args[0], rate)


AUGMENTATIONS = {
    "freq_masking":     FreqMasking,
    "time_masking":     TimeMasking,
    "noise":            Noise,
    "white_noise":      WhiteNoise,
    "real_world_noise": RealWorldNoise,
    "time_stretch":     TimeStretch
}


class UserAugmentation(UserDict):
    def __init__(self, config: dict):
        config["before"] = self.parse(config.get("before", {}))
        config["after"] = self.parse(config.get("after", {}))
        config["include_original"] = config.get("include_original", False)
        super(UserAugmentation, self).__init__(config)

    @staticmethod
    def parse(config: dict) -> list:
        augmentations = []
        for key, value in config.items():
            if not AUGMENTATIONS.get(key, None):
                raise KeyError(f"No augmentation named: {key}\n"
                               f"Available augmentations: {AUGMENTATIONS.keys()}")
            augmentations.append(AUGMENTATIONS[key](value))
        return augmentations
