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
    def __init__(self, func, is_post=True):
        self.func = func
        # Whether postaugmentation or preaugmentation of feature extraction
        self.is_post = is_post

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class FreqMasking(Augmentation):
    def __init__(self, num_freq_mask: int = 1, freq_mask_param: int = 10):
        self.num_freq_mask = num_freq_mask
        self.freq_mask_param = freq_mask_param
        super(FreqMasking, self).__init__(
            func=functools.partial(freq_masking, num_freq_mask=self.num_freq_mask,
                                   freq_mask_param=self.freq_mask_param),
            is_post=True
        )


class TimeMasking(Augmentation):
    def __init__(self, num_time_mask: int = 1, time_mask_param: int = 50,
                 p_upperbound: float = 1.0):
        self.num_time_mask = num_time_mask
        self.time_mask_param = time_mask_param
        self.p_upperbound = p_upperbound
        super(TimeMasking, self).__init__(
            func=functools.partial(time_masking, num_time_mask=self.num_time_mask,
                                   time_mask_param=self.time_mask_param, p_upperbound=self.p_upperbound),
            is_post=True
        )


class TimeWarping(Augmentation):
    def __init__(self, time_warp_param: int = 50):
        self.time_warp_param = time_warp_param
        super(TimeWarping, self).__init__(
            func=functools.partial(time_warping, time_warp_param=self.time_warp_param),
            is_post=True
        )


class Noise(Augmentation):
    def __init__(self, noise_dir: str, snr_list: list = (0, 5, 10, 15), max_noises: int = 3):
        self.snr_list = list(snr_list)
        if not any([i < 0 for i in self.snr_list]):
            self.snr_list.append(-1)
        self.max_noises = max_noises
        self.noises = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
        self.noises.append("white_noise")
        super(Noise, self).__init__(
            func=functools.partial(add_noise, snr_list=self.snr_list, max_noises=self.max_noises, noises=self.noises),
            is_post=False
        )


class WhiteNoise(Augmentation):
    def __init__(self, snr_list: list = (0, 5, 10, 15)):
        self.snr_list = list(snr_list)
        if not any([i < 0 for i in self.snr_list]):
            self.snr_list.append(-1)
        super(WhiteNoise, self).__init__(
            func=functools.partial(add_white_noise, snr_list=self.snr_list),
            is_post=False
        )


class RealWorldNoise(Augmentation):
    def __init__(self, noise_dir: str, snr_list: list = (0, 5, 10, 15), max_noises: int = 3):
        self.snr_list = list(snr_list)
        if not any([i < 0 for i in self.snr_list]):
            self.snr_list.append(-1)
        self.max_noises = max_noises
        self.noises = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
        super(RealWorldNoise, self).__init__(
            func=functools.partial(add_realworld_noise, snr_list=self.snr_list, max_noises=self.max_noises, noises=self.noises),
            is_post=False
        )


class TimeStretch(Augmentation):
    def __init__(self, min_ratio: float = 0.5, max_ratio: float = 2.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        super(TimeStretch, self).__init__(func=self.func, is_post=False)

    def func(self, signal, **kwargs):
        rate = random.uniform(self.min_ratio, self.max_ratio)
        return librosa.effects.time_stretch(signal, rate)
