from __future__ import absolute_import

import functools
import os
import glob
import librosa
import random
from augmentations.SpecAugment import time_warping, time_masking, \
    freq_masking
from augmentations.NoiseAugment import add_noise, add_white_noise, add_realworld_noise


class Augmentation:
    def __init__(self, func, is_post=True, **kwargs):
        self.func = func
        # Whether postaugmentation or preaugmentation of feature extraction
        self.is_post = is_post
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self.kwargs:
            return self.func(*args, **self.kwargs)
        return self.func(*args, **kwargs)


class FreqMasking(Augmentation):
    def __init__(self, **kwargs):
        super(FreqMasking, self).__init__(func=freq_masking, is_post=True, **kwargs)


class TimeMasking(Augmentation):
    def __init__(self, **kwargs):
        super(TimeMasking, self).__init__(func=time_masking, is_post=True, **kwargs)


class TimeWarping(Augmentation):
    def __init__(self, **kwargs):
        super(TimeWarping, self).__init__(func=time_warping, is_post=True, **kwargs)


class Noise(Augmentation):
    def __init__(self, snr_list: list, min_noises: int, max_noises: int, noise_dir: str):
        self.snr_list = snr_list
        if not any([i == 0 for i in self.snr_list]):
            self.snr_list.append(0)
        self.min_noises = min_noises
        self.max_noises = max_noises
        self.noises = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
        self.noises.append("white_noise")
        super(Noise, self).__init__(
            func=functools.partial(add_noise, snr_list=self.snr_list, min_noises=self.min_noises,
                                   max_noises=self.max_noises, noises=self.noises),
            is_post=False
        )


class WhiteNoise(Augmentation):
    def __init__(self, snr_list: list):
        self.snr_list = snr_list
        if not any([i == 0 for i in self.snr_list]):
            self.snr_list.append(0)
        super(WhiteNoise, self).__init__(
            func=functools.partial(add_white_noise, snr_list=self.snr_list),
            is_post=False
        )


class RealWorldNoise(Augmentation):
    def __init__(self, snr_list: list, min_noises: int, max_noises: int, noise_dir: str):
        self.snr_list = snr_list
        if not any([i == 0 for i in self.snr_list]):
            self.snr_list.append(0)
        self.min_noises = min_noises
        self.max_noises = max_noises
        self.noises = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
        super(RealWorldNoise, self).__init__(
            func=functools.partial(add_realworld_noise, snr_list=self.snr_list, min_noises=self.min_noises,
                                   max_noises=self.max_noises, noises=self.noises),
            is_post=False
        )


class TimeStretch(Augmentation):
    def __init__(self, min_ratio: float = 1.0, max_ratio: float = 1.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        super(TimeStretch, self).__init__(func=self.func, is_post=False)

    def func(self, signal, **kwargs):
        rate = random.uniform(self.min_ratio, self.max_ratio)
        return librosa.effects.time_stretch(signal, rate)
