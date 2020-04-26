from __future__ import absolute_import

import functools
import numpy as np
from augmentations.SpecAugment import time_warping, time_masking, \
  freq_masking
from augmentations.NoiseAugment import add_white_noise, add_noise_from_sound


def no_aug(features):
  return features


class Augmentation:
  def __init__(self, func, is_post=True, **kwargs):
    self.func = func
    # Whether postaugmentation or preaugmentation of feature
    # extraction
    self.is_post = is_post
    self.kwargs = kwargs  # Save parameters in config

  def __call__(self, *args, **kwargs):
    if self.kwargs:
      return self.func(*args, **self.kwargs)
    return self.func(*args, **kwargs)


class FreqMasking(Augmentation):
  def __init__(self, **kwargs):
    super().__init__(func=freq_masking, is_post=True, **kwargs)


class TimeMasking(Augmentation):
  def __init__(self, **kwargs):
    super().__init__(func=time_masking, is_post=True, **kwargs)


class TimeWarping(Augmentation):
  def __init__(self, **kwargs):
    super().__init__(func=time_warping, is_post=True, **kwargs)


class WhiteNoise(Augmentation):
  def __init__(self, snr=10, **kwargs):
    self.snr = snr
    super().__init__(func=functools.partial(add_white_noise, snr=self.snr), is_post=False, **kwargs)


class RealWorldNoise(Augmentation):
  def __init__(self, noise_wavs: list, snr=10, **kwargs):
    if not noise_wavs or len(noise_wavs) == 0:
      raise ValueError("List of noise wav files must be defined")
    self.noise_wavs = noise_wavs
    self.snr = snr
    super(RealWorldNoise, self).__init__(
      func=functools.partial(add_noise_from_sound, noise_wavs=self.noise_wavs, snr=self.snr),
      is_post=False, **kwargs)
