from __future__ import absolute_import

import math
import os
import io
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf


def compute_spectrogram_feature(signal, sample_rate, frame_ms, stride_ms, num_feature_bins):
  frame_length = int(sample_rate * (frame_ms / 1000))
  frame_step = int(sample_rate * (stride_ms / 1000))

  powspec = np.square(
    np.abs(
      librosa.core.stft(
        signal, n_fft=frame_length, hop_length=frame_step,
        win_length=frame_length, center=True
      )))

  # remove small bins
  powspec[powspec <= 1e-30] = 1e-30
  features = 10 * np.log10(powspec.T)

  assert num_feature_bins <= frame_length // 2 + 1, \
    "num_features for spectrogram should \
      be <= (sample_rate * window_size // 2 + 1)"

  # cut high frequency part, keep num_feature_bins features
  features = features[:, :num_feature_bins]

  return features


def compute_mfcc_feature(signal, sample_rate, frame_ms, stride_ms, num_feature_bins):
  frame_length = int(sample_rate * (frame_ms / 1000))
  frame_step = int(sample_rate * (stride_ms / 1000))
  num_fft = 2 ** math.ceil(math.log2(frame_length))

  signal = preemphasis(signal, coeff=0.97)
  S = np.square(
    np.abs(
      librosa.core.stft(
        signal, n_fft=num_fft, hop_length=frame_length,
        win_length=frame_step, center=True
      )))

  return librosa.feature.mfcc(sr=sample_rate, S=S,
                              n_mfcc=num_feature_bins,
                              n_mels=2 * num_feature_bins).T


def compute_logfbank_feature(signal, sample_rate, frame_ms, stride_ms, num_feature_bins):
  frame_length = int(sample_rate * (frame_ms / 1000))
  frame_step = int(sample_rate * (stride_ms / 1000))
  num_fft = 2 ** math.ceil(math.log2(frame_length))

  signal = preemphasis(signal, coeff=0.97)
  S = np.square(
    np.abs(
      librosa.core.stft(
        signal, n_fft=num_fft, hop_length=frame_length,
        win_length=frame_step, center=True
      )))

  mel_basis = librosa.filters.mel(sample_rate, num_fft,
                                  n_mels=num_feature_bins,
                                  fmin=0, fmax=int(sample_rate / 2))

  return np.log(np.dot(mel_basis, S) + 1e-20).T


def read_raw_audio(audio, sample_rate=16000):
  if isinstance(audio, str):
    wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
  elif isinstance(audio, bytes):
    wave, sr = sf.read(io.BytesIO(audio))
    if sr != sample_rate:
      wave = librosa.resample(wave, sr, sample_rate)
  else:
    raise ValueError("input audio must be either a path or bytes")
  return wave


def normalize_audio_feature(audio_feature):
  """ Mean and variance normalization """
  mean = np.mean(audio_feature, axis=0)
  std_dev = np.std(audio_feature, axis=0)
  normalized = (audio_feature - mean) / std_dev
  return normalized


def normalize_signal(signal):
  """Normailize signal to [-1, 1] range"""
  gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
  return signal * gain


def preemphasis(signal, coeff=0.97):
  if not coeff or coeff == 0.0:
    return signal
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def deemphasis(signal, coeff=0.97):
  if coeff <= 0:
    return signal
  x = np.zeros(signal.shape[0], dtype=np.float32)
  x[0] = signal[0]
  for n in range(1, signal.shape[0], 1):
    x[n] = coeff * x[n - 1] + signal[n]
  return x


class SpeechFeaturizer(tf.keras.layers.Layer):
  """A class for extraction speech features"""

  def __init__(self, sample_rate, frame_ms, stride_ms, num_feature_bins,
               feature_type="mfcc", name="speech_featurizer", **kwargs):
    self.sample_rate = sample_rate
    self.num_feature_bins = num_feature_bins
    self.feature_type = feature_type
    self.frame_length = int(self.sample_rate * (frame_ms / 1000))
    self.frame_step = int(self.sample_rate * (stride_ms / 1000))
    self.num_fft = 2 ** math.ceil(math.log2(self.frame_length))
    super(SpeechFeaturizer, self).__init__(name=name, trainable=False, **kwargs)

  def __compute_spectrogram(self, signal):
    return tf.abs(tf.signal.stft(signal, frame_length=self.frame_length,
                                 frame_step=self.frame_step, fft_length=self.num_fft))

  def __compute_logfbank(self, signal):
    spectrogram = self.__compute_spectrogram(signal)
    linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=self.num_feature_bins,
      num_spectrogram_bins=spectrogram.shape[-1],
      sample_rate=self.sample_rate,
      lower_edge_hertz=80.0, upper_edge_hertz=7600.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_weight_matrix.shape[-1:]))
    return tf.math.log(mel_spectrogram + 1e-6)

  def compute_tf_spectrogram_features(self, signal):
    spectrogram = self.__compute_spectrogram(signal)
    spectrogram = spectrogram[:, :self.num_feature_bins]
    return tf.expand_dims(spectrogram, -1)

  def compute_tf_logfbank_features(self, signal):
    log_mel_spectrogram = self.__compute_logfbank(signal)
    log_mel_spectrogram = log_mel_spectrogram[:, :self.num_feature_bins]
    return tf.expand_dims(log_mel_spectrogram, -1)

  def compute_tf_mfcc_features(self, signal):
    log_mel_spectrogram = self.__compute_logfbank(signal)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    return tf.expand_dims(mfcc, -1)

  def call(self, signal, **kwargs):
    if self.feature_type == "mfcc":
      return self.compute_tf_mfcc_features(signal)
    elif self.feature_type == "spectrogram":
      return self.compute_tf_spectrogram_features(signal)
    elif self.feature_type == "logfbank":
      return self.compute_tf_logfbank_features(signal)

  def get_config(self):
    config = super(SpeechFeaturizer, self).get_config()
    config.update({'sample_rate':      self.sample_rate,
                   'num_feature_bins': self.num_feature_bins,
                   'frame_length':     self.frame_length,
                   'frame_step':       self.frame_step,
                   'num_fft':          self.num_fft})
    return config

  def from_config(self, config):
    return self(**config)
