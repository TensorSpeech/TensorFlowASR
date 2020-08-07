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
import glob
import os
from collections import UserDict

import librosa
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
from nlpaug.util import Action
from nlpaug.augmenter.spectrogram import SpectrogramAugmenter

from .spec_augment import FreqMaskingModel, TimeMaskingModel


class SignalCropping(naa.CropAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=0.1,
                 crop_range=(0.2, 0.8),
                 crop_factor=2):
        super(SignalCropping, self).__init__(sampling_rate=None, zone=zone, coverage=coverage,
                                             crop_range=crop_range, crop_factor=crop_factor,
                                             duration=None)


class SignalLoudness(naa.LoudnessAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(0.5, 2)):
        super(SignalLoudness, self).__init__(zone=zone, coverage=coverage, factor=factor)


class SignalMask(naa.MaskAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 mask_range=(0.2, 0.8),
                 mask_factor=2,
                 mask_with_noise=True):
        super(SignalMask, self).__init__(sampling_rate=None, zone=zone, coverage=coverage,
                                         duration=None, mask_range=mask_range,
                                         mask_factor=mask_factor,
                                         mask_with_noise=mask_with_noise)


class SignalNoise(naa.NoiseAug):
    def __init__(self,
                 sample_rate=16000,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 color="random",
                 noises: str = None):
        if noises is not None:
            noises = glob.glob(os.path.join(noises, "**", "*.wav"), recursive=True)
            noises = [librosa.load(n, sr=sample_rate)[0] for n in noises]
        super(SignalNoise, self).__init__(zone=zone, coverage=coverage,
                                          color=color, noises=noises)


class SignalPitch(naa.PitchAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(-10, 10)):
        super(SignalPitch, self).__init__(None, zone=zone, coverage=coverage,
                                          duration=None, factor=factor)


class SignalShift(naa.ShiftAug):
    def __init__(self,
                 sample_rate=16000,
                 duration=3,
                 direction="random"):
        super(SignalShift, self).__init__(sample_rate, duration=duration, direction=direction)


class SignalSpeed(naa.SpeedAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(0.5, 2)):
        super(SignalSpeed, self).__init__(zone=zone, coverage=coverage,
                                          duration=None, factor=factor)


class SignalVtlp(naa.VtlpAug):
    def __init__(self,
                 sample_rate=16000,
                 zone=(0.2, 0.8),
                 coverage=0.1,
                 fhi=4800,
                 factor=(0.9, 1.1)):
        super(SignalVtlp, self).__init__(sample_rate, zone=zone, coverage=coverage,
                                         duration=None, fhi=fhi, factor=factor)

# class SeganAugment(Augmentation):
#     def __init__(self, params: dict):
#         with tf.device("/device:CPU:0"):
#             self.segan_inferencer = SeganTFLite(
#                 speech_config=params["speech_config"],
#                 saved_path=params["saved_path"]
#             )
#             self.segan_inferencer.compile()
#         super(SeganAugment, self).__init__({})
#
#     def func(self, *args, **kwargs):
#         if random.choice([0, 1]) == 0: return kwargs["signal"]
#         with tf.device("/device:CPU:0"):  # keep gpu for training main model
#             gen_signal = self.segan_inferencer.infer(kwargs["signal"])
#         if self.segan_inferencer.speech_config["sample_rate"] != kwargs["sample_rate"]:
#             gen_signal = librosa.resample(
#                 gen_signal,
#                 self.segan_inferencer.speech_config["sample_rate"],
#                 kwargs["sample_rate"]
#             )
#         return gen_signal


class TimeMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: int = 100,
                 p_upperbound: float = 0.05,
                 name="TimeMasking",
                 verbose=0):
        super(TimeMasking, self).__init__(
            action=Action.SUBSTITUTE, name=name, device="cpu", verbose=verbose)
        self.model = self.get_model(num_masks, mask_factor, p_upperbound)

    def substitute(self, data):
        return self.model.mask(data)

    @classmethod
    def get_model(cls,
                  num_masks,
                  mask_factor,
                  p_upperbound):
        return TimeMaskingModel(num_masks, mask_factor, p_upperbound)


class FreqMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: int = 27,
                 name="FreqMasking",
                 verbose=0):
        super(FreqMasking, self).__init__(
            action=Action.SUBSTITUTE, name=name, device="cpu", verbose=verbose)
        self.model = self.get_model(num_masks, mask_factor)

    def substitute(self, data):
        return self.model.mask(data)

    @classmethod
    def get_model(cls,
                  num_masks,
                  mask_factor):
        return FreqMaskingModel(num_masks, mask_factor)


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
        config["before"] = self.parse(
            config.get("before", {}).get("methods", {}),
            sometimes=config.get("before", {}).get("sometimes", True)
        )
        config["after"] = self.parse(
            config.get("after", {}).get("methods", {}),
            sometimes=config.get("after", {}).get("sometimes", True)
        )
        config["include_original"] = config.get("include_original", False)
        super(UserAugmentation, self).__init__(config)

    def __missing__(self, key):
        return None

    @staticmethod
    def parse(config: dict, sometimes: bool = True) -> list:
        augmentations = []
        for key, value in config.items():
            if not AUGMENTATIONS.get(key, None):
                raise KeyError(f"No augmentation named: {key}\n"
                               f"Available augmentations: {AUGMENTATIONS.keys()}")
            aug = AUGMENTATIONS[key](**value) if value is not None else AUGMENTATIONS[key]()
            augmentations.append(aug)
        if not sometimes:
            return naf.Sequential(augmentations)
        return naf.Sometimes(augmentations)
