from __future__ import absolute_import

import runpy
import os
import numpy as np
import tensorflow as tf
from nltk.metrics import distance
from configs import DefaultConfig, SeganConfig

asr_conf_required = ["base_model",
                     "decoder",
                     "batch_size",
                     "num_epochs",
                     "vocabulary_file_path",
                     "learning_rate",
                     "min_lr",
                     "sample_rate",
                     "frame_ms",
                     "stride_ms",
                     "num_feature_bins",
                     "feature_type",
                     "streaming_size"]

asr_conf_paths = ["train_data_transcript_paths",
                  "test_data_transcript_paths",
                  "eval_data_transcript_paths",
                  "vocabulary_file_path",
                  "checkpoint_dir",
                  "log_dir"]

segan_conf_required = ["batch_size",
                       "num_epochs",
                       "kwidth",
                       "ratio",
                       "noise_std",
                       "denoise_epoch",
                       "noise_decay",
                       "noise_std_lbound",
                       "l1_lambda",
                       "pre_emph",
                       "window_size",
                       "stride",
                       "g_learning_rate",
                       "d_learning_rate"]

segan_conf_paths = ["clean_train_data_dir",
                    "noisy_train_data_dir",
                    "clean_test_data_dir",
                    "noisy_test_data_dir",
                    "checkpoint_dir",
                    "log_dir"]


def check_key_in_dict(dictionary, keys):
  for key in keys:
    if key not in dictionary.keys():
      raise ValueError("{} must be defined".format(key))


def preprocess_paths(paths):
  if isinstance(paths, list):
    return list(map(os.path.expanduser, paths))
  return os.path.expanduser(paths)


def get_asr_config(config_path):
  conf_dict = runpy.run_path(config_path)
  check_key_in_dict(dictionary=conf_dict, keys=asr_conf_required)
  # fill missing default optional values
  default_dict = vars(DefaultConfig)
  for key in default_dict.keys():
    if key not in conf_dict.keys():
      conf_dict[key] = default_dict[key]
  # convert paths to take ~/ dir
  for key in asr_conf_paths:
    conf_dict[key] = preprocess_paths(conf_dict[key])

  return conf_dict


def get_segan_config(config_path):
  conf_dict = runpy.run_path(config_path)
  check_key_in_dict(dictionary=conf_dict, keys=segan_conf_required)
  # fill missing default optional values
  default_dict = vars(SeganConfig)
  for key in default_dict.keys():
    if key not in conf_dict.keys():
      conf_dict[key] = default_dict[key]
  # convert paths to take ~/ dir
  for key in segan_conf_paths:
    conf_dict[key] = preprocess_paths(conf_dict[key])

  return conf_dict


def levenshtein(a, b):
  """
  Calculate Levenshtein distance between a and b
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n
  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)
  return current[n]


def wer(decode, target):
  words = set(decode.split() + target.split())
  word2char = dict(zip(words, range(len(words))))

  new_decode = [chr(word2char[w]) for w in decode.split()]
  new_target = [chr(word2char[w]) for w in target.split()]

  return distance.edit_distance(''.join(new_decode),
                                ''.join(new_target)), len(target.split())


def mywer(decode, target):
  dist = levenshtein(target.lower().split(), decode.lower().split())
  return dist, len(target.split())


def cer(decode, target):
  return distance.edit_distance(decode, target), len(target)


def mask_nan(x):
  x_zeros = tf.zeros_like(x)
  x_mask = tf.math.is_finite(x)
  y = tf.where(x_mask, x, x_zeros)
  return y


def bytes_to_string(array, encoding: str = "utf-8"):
  return [transcript.decode(encoding) for transcript in array]


def get_length(batch_data):
  def map_fn(elem):
    size = tf.shape(elem)
    return tf.convert_to_tensor(size[0], dtype=tf.int32)

  return tf.map_fn(map_fn, batch_data, dtype=tf.int32)


def slice_signal(signal, window_size, stride=0.5):
  """ Return windows of the given signal by sweeping in stride fractions
      of window
  """
  assert signal.ndim == 1, signal.ndim
  n_samples = signal.shape[0]
  offset = int(window_size * stride)
  slices = []
  for beg_i, end_i in zip(range(0, n_samples, offset),
                          range(window_size, n_samples + offset,
                                offset)):
    if end_i - beg_i < window_size:
      break
    slice_ = signal[beg_i:end_i]
    # if slice_.shape[0] < window_size:
    #   slice_ = np.pad(slice_, (0, window_size - slice_.shape[0]), 'constant', constant_values=0.0)
    if slice_.shape[0] == window_size:
      slices.append(slice_)
  return np.array(slices, dtype=np.float32)


def merge_slices(slices):
  # slices shape = [batch, window_size]
  return tf.keras.backend.flatten(slices)  # return shape = [-1, ]


@tf.function
def scalar_summary(name, x):
  return tf.summary.scalar(name, x)


@tf.function
def histogram_summary(name, x):
  return tf.summary.histogram(name, x)


@tf.function
def audio_summary(name, x, samplerate):
  return tf.summary.audio(name, x, samplerate)
