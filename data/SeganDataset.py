from __future__ import absolute_import

import glob
import os
import librosa
import tensorflow as tf
from utils.Utils import slice_signal
from featurizers.SpeechFeaturizer import preemphasis


class SeganDataset:
  def __init__(self, clean_data_dir, noisy_data_dir, window_size=2 ** 14, stride=0.5):
    self.clean_data_dir = clean_data_dir
    self.noisy_data_dir = noisy_data_dir
    self.window_size = window_size
    self.stride = stride

  def create(self, batch_size, coeff=0.97, repeat=1):
    def _gen_data():
      for clean_wav_path in glob.iglob(os.path.join(self.clean_data_dir, "**", "*.wav"), recursive=True):
        clean_split = clean_wav_path.split('/')
        noisy_split = self.noisy_data_dir.split('/')
        clean_split = clean_split[len(noisy_split):]
        noisy_split = noisy_split + clean_split
        noisy_wav_path = '/' + os.path.join(*noisy_split)

        clean_wav, clean_sr = librosa.load(clean_wav_path, sr=None)
        noisy_wav, noisy_sr = librosa.load(noisy_wav_path, sr=None)
        assert clean_sr == 16000 and noisy_sr == 16000, "sample rate of dataset must be 16k"
        clean_slices = slice_signal(clean_wav, self.window_size, self.stride)
        noisy_slices = slice_signal(noisy_wav, self.window_size, self.stride)

        for clean_slice, noisy_slice in zip(clean_slices, noisy_slices):
          if len(clean_slice) == 0:
            continue
          yield preemphasis(clean_slice, coeff), preemphasis(noisy_slice, coeff)

    dataset = tf.data.Dataset.from_generator(
      _gen_data,
      output_types=(
        tf.float32,
        tf.float32
      ),
      output_shapes=(
        tf.TensorShape([self.window_size]),
        tf.TensorShape([self.window_size])
      )
    )
    # Repeat and batch the dataset
    dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    # Prefetch to improve speed of input length
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
