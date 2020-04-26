from __future__ import absolute_import

import math
import os
import io
import numpy as np
import librosa
import tensorflow as tf

WINDOW_FN = {"hanning": np.hanning, "hamming": np.hamming}


class MFCC(tf.keras.layers.Layer):
  def __init__(self, name, sample_rate, frame_ms, stride_ms, num_feature_bins, **kwargs):
    self.sample_rate = sample_rate
    self.num_feature_bins = num_feature_bins
    self.frame_length = int(self.sample_rate * (frame_ms / 1000))
    self.frame_step = int(self.sample_rate * (stride_ms / 1000))
    self.num_fft = 2 ** math.ceil(math.log2(self.frame_length))
    super(MFCC, self).__init__(name=name, **kwargs)

  def call(self, signal, **kwargs):
    spectrogram = tf.abs(tf.signal.stft(signal, frame_length=self.frame_length,
                                        frame_step=self.frame_step, fft_length=self.num_fft))
    linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=self.num_feature_bins,
      num_spectrogram_bins=spectrogram.shape[-1],
      sample_rate=self.sample_rate,
      lower_edge_hertz=80.0, upper_edge_hertz=7600.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    return tf.expand_dims(mfcc, -1)

  def get_config(self):
    config = super(MFCC, self).get_config()
    config.update({'sample_rate':      self.sample_rate,
                   'num_feature_bins': self.num_feature_bins,
                   'frame_length':     self.frame_length,
                   'frame_step':       self.frame_step,
                   'num_fft':          self.num_fft})
    return config

  def from_config(self, config):
    return self(**config)


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
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def deemphasis(signal, coeff=0.97):
  if coeff <= 0:
    return signal
  x = np.zeros(signal.shape[0], dtype=np.float32)
  x[0] = signal[0]
  for n in range(1, signal.shape[0], 1):
    x[n] = coeff * x[n - 1] + signal[n]
  return x


class SpeechFeaturizer:
  """A class for extraction speech features"""

  def __init__(self, sample_rate, frame_ms, stride_ms, num_feature_bins,
               window_fn=WINDOW_FN.get("hanning"), feature_type="mfcc"):
    self.sample_rate = sample_rate
    self.frame_ms = frame_ms
    self.stride_ms = stride_ms
    self.num_feature_bins = num_feature_bins
    self.window_fn = window_fn
    self.feature_type = feature_type
    self.n_window_size = int(self.sample_rate * (self.frame_ms / 1000))
    self.n_window_stride = int(self.sample_rate * (self.stride_ms / 1000))
    self.num_fft = 2 ** math.ceil(math.log2(self.n_window_size))

  def __compute_spectrogram_feature(self, signal):
    """Function to convert raw audio signal to spectrogram using
    librosa backend
    Args:
        signal (np.array): np.array containing raw audio signal
    Returns:
        features (np.array): spectrogram of shape=[num_timesteps,
        num_feature_bins]
        audio_duration (float): duration of the signal in seconds
    """

    powspec = np.square(
      np.abs(
        librosa.core.stft(
          signal, n_fft=self.n_window_size,
          hop_length=self.n_window_stride, win_length=self.n_window_size,
          center=True, window=self.window_fn
        )))

    # remove small bins
    powspec[powspec <= 1e-30] = 1e-30
    features = 10 * np.log10(powspec.T)

    assert self.num_feature_bins <= self.n_window_size // 2 + 1, \
      "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

    # cut high frequency part, keep num_feature_bins features
    features = features[:, :self.num_feature_bins]

    return features

  def __compute_mfcc_feature(self, signal):
    """Function to convert raw audio signal to mfcc using
    librosa backend
    Args:
        signal (np.array): np.array containing raw audio signal
    Returns:
        features (np.array): mfcc of shape=[num_timesteps,
        num_feature_bins]
        audio_duration (float): duration of the signal in seconds
    """

    signal = preemphasis(signal, coeff=0.97)
    S = np.square(
      np.abs(
        librosa.core.stft(
          signal, n_fft=self.num_fft,
          hop_length=self.n_window_stride,
          win_length=self.n_window_size,
          center=True, window=self.window_fn
        )))

    return librosa.feature.mfcc(sr=self.sample_rate, S=S,
                                n_mfcc=self.num_feature_bins,
                                n_mels=2 * self.num_feature_bins).T

  def __compute_logfbank_feature(self, signal):
    """Function to convert raw audio signal to logfbank using
    librosa backend
    Args:
        signal (np.array): np.array containing raw audio signal
    Returns:
        features (np.array): mfcc of shape=[num_timesteps,
        num_feature_bins]
        audio_duration (float): duration of the signal in seconds
    """
    num_fft = 2 ** math.ceil(math.log2(self.n_window_size))

    signal = preemphasis(signal, coeff=0.97)
    S = np.square(
      np.abs(
        librosa.core.stft(
          signal, n_fft=num_fft,
          hop_length=self.n_window_stride,
          win_length=self.n_window_size,
          center=True, window=self.window_fn
        )))

    mel_basis = librosa.filters.mel(self.sample_rate, num_fft,
                                    n_mels=self.num_feature_bins,
                                    fmin=0, fmax=int(self.sample_rate / 2))

    return np.log(np.dot(mel_basis, S) + 1e-20).T

  @staticmethod
  def convert_bytesarray_to_float(bytesarray, channels=2):
    # 16-bit little-endian requires 2 bytes to construct 32-bit float
    bytesarray = librosa.util.buf_to_float(bytesarray, n_bytes=2)
    if channels == 2:
      bytesarray = bytesarray.reshape((-1, channels)).T
    return librosa.core.to_mono(bytesarray)

  def compute_speech_features(self, audio_file_path, sr=44100, channels=2):
    """Load audio file, preprocessing, compute features,
    postprocessing
    Args:
        audio_file_path (string or np.array): the path to audio file
        or audio data
        sr (int): default sample rate
        channels: default channels
    Returns:
        features (np.array): spectrogram of shape=[num_timesteps,
        num_feature_bins, 1]
        audio_duration (float): duration of the signal in seconds
    """
    if isinstance(audio_file_path, str):
      data, sr = librosa.core.load(os.path.expanduser(audio_file_path), sr=None)
    elif isinstance(audio_file_path, bytes):
      data = self.convert_bytesarray_to_float(audio_file_path,
                                              channels=channels)
    elif isinstance(audio_file_path, tf.Tensor):
      data = audio_file_path
    else:
      raise ValueError(
        "audio_file_path must be string, bytes or tf.Tensor")

    if sr != self.sample_rate:
      data = librosa.core.resample(
        data, orig_sr=sr, target_sr=self.sample_rate, scale=True)

    data = normalize_signal(data.astype(np.float32))
    if self.feature_type == "mfcc":
      data = self.__compute_mfcc_feature(data)
    elif self.feature_type == "spectrogram":
      data = self.__compute_spectrogram_feature(data)
    elif self.feature_type == "logfbank":
      data = self.__compute_logfbank_feature(data)
    else:
      raise ValueError("feature_type must be either 'mfcc', 'spectrogram' \
                       or 'logfbank'")
    data = normalize_audio_feature(data)

    # Adding Channel dimmension for conv2D input
    data = np.expand_dims(data, axis=2)
    return tf.convert_to_tensor(data, dtype=tf.float32)

  def read_raw_audio(self, audio):
    if isinstance(audio, str):
      wave, _ = librosa.load(os.path.expanduser(audio), sr=self.sample_rate)
    elif isinstance(audio, bytes):
      wave, _ = librosa.load(io.BytesIO(audio), sr=self.sample_rate)
    else:
      raise ValueError("input audio must be either a path or bytes")
    return tf.convert_to_tensor(wave, dtype=tf.float32)
