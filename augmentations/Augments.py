from __future__ import absolute_import

import functools
import os
import glob
from augmentations.SpecAugment import time_warping, time_masking, \
  freq_masking
from augmentations.NoiseAugment import add_noise


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


class Noise(Augmentation):
  def __init__(self, min_snr: int, max_snr: int, min_noises: int, max_noises: int, noise_dir: str, **kwargs):
    self.min_snr = min_snr
    self.max_snr = max_snr
    self.min_noises = min_noises
    self.max_noises = max_noises
    self.noises = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
    self.noises.append("white_noise")
    super().__init__(func=functools.partial(
      add_noise, min_snr=self.min_snr,
      max_snr=self.max_snr, min_noises=self.min_noises,
      max_noises=self.max_noises, noises=self.noises), is_post=False, **kwargs)
