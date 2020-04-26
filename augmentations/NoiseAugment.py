from __future__ import absolute_import

import math
import numpy as np
import random
from featurizers.SpeechFeaturizer import read_raw_audio


def add_white_noise(signal: np.ndarray, snr=10, sample_rate=16000):
  RMS_s = math.sqrt(np.mean(signal ** 2))
  # RMS values of noise
  RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 20)))
  # Additive white gausian noise. Thereore mean=0
  # Because sample length is large (typically > 40000)
  # we can use the population formula for standard daviation.
  # because mean=0 STD=RMS
  STD_n = RMS_n
  noise = np.random.normal(0, STD_n, signal.shape[0])
  return signal + noise


def add_noise_from_sound(signal: np.ndarray, noise_wavs: list, snr=10, sample_rate=16000):
  noise = random.choice(noise_wavs)  # randomly choose a noise from a list of noises
  noise = read_raw_audio(noise, sample_rate)

  if len(noise) < len(signal):
    raise ValueError("Noise wav must be longer than speech")
  else:
    idx = random.choice(range(0, len(noise) - len(signal)))  # randomly crop noise wav
    noise = noise[idx:idx + len(signal)]

  RMS_s = math.sqrt(np.mean(signal ** 2))
  # required RMS of noise
  RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 20)))

  # current RMS of noise
  RMS_n_current = math.sqrt(np.mean(noise ** 2))
  noise = noise * (RMS_n / RMS_n_current)

  return noise
