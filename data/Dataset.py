from __future__ import absolute_import

import functools
import multiprocessing
import numpy as np
import tensorflow as tf
from featurizers.SpeechFeaturizer import read_raw_audio, preemphasis


class Dataset:
  def __init__(self, data_path, mode="train"):
    self.data_path = data_path
    self.mode = mode
    self.num_cpus = multiprocessing.cpu_count()

  def __call__(self, text_featurizer, sample_rate=16000, preemph=0.95, batch_size=32,
               repeat=1, augmentations=tuple([None]), sortagrad=False):
    entries = self.__create_entries(augmentations, sortagrad)
    if self.mode == "train":
      return self.__create_dataset(entries=entries, text_featurizer=text_featurizer,
                                   sample_rate=sample_rate, preemph=preemph,
                                   sort=sortagrad, shuffle=True, augmentations=augmentations,
                                   batch_size=batch_size, repeat=repeat)
    if self.mode in ["eval", "test", "infer"]:
      return self.__create_dataset(entries=entries, text_featurizer=text_featurizer,
                                   sample_rate=sample_rate, preemph=preemph, augmentations=[None],
                                   sort=sortagrad, shuffle=False, batch_size=batch_size)
    raise ValueError("Mode must be 'train', 'eval' or 'infer'")

  @staticmethod
  def entries_map_fn(splitted_lines, augmentations):
    results = []
    for path, _, transcript in splitted_lines:
      for idx, au in enumerate(augmentations):
        results.append([path, idx, transcript])
    return np.array(results)

  def __create_entries(self, augmentations, sort=False):
    lines = []
    for file_path in self.data_path:
      with tf.io.gfile.GFile(file_path, "r") as f:
        temp_lines = f.read().splitlines()
        # Skip the header of csv file
        lines += temp_lines[1:]
    # The files is "\t" seperated
    lines = [line.split("\t", 2) for line in lines]
    if sort:
      lines.sort(key=lambda item: int(item[1]))
    lines = np.array(lines)
    splitted_lines = np.array_split(lines, self.num_cpus)
    with multiprocessing.Pool(self.num_cpus) as pool:
      lines = pool.map(functools.partial(self.entries_map_fn, augmentations=augmentations), splitted_lines)
    return np.concatenate(lines)

  @staticmethod
  def preprocess(audio_file, au_idx, transcript, text_featurizer=None,
                 augmentations=None, sample_rate=16000, coeff=None):
    signal = read_raw_audio(audio_file.numpy().decode("utf-8"), sample_rate)
    labels = text_featurizer.compute_label_features(transcript.numpy().decode("utf-8"))
    if augmentations[au_idx] is not None:
      signal = augmentations[au_idx](signal=signal, sample_rate=sample_rate)
    signal = preemphasis(signal, coeff=coeff)
    label_length = tf.cast(tf.shape(labels)[0], tf.int32)
    return tf.convert_to_tensor(signal, tf.float32), labels, label_length

  @staticmethod
  def preprocess_map_fn(audio_file, au_idx, transcript, text_featurizer=None,
                        augmentations=None, sample_rate=16000, coeff=None):
    signal, labels, label_length = tf.py_function(
      functools.partial(Dataset.preprocess, text_featurizer=text_featurizer,
                        augmentations=augmentations, sample_rate=sample_rate, coeff=coeff),
      inp=[audio_file, au_idx, transcript],
      Tout=(tf.float32, tf.int32, tf.int32))
    signal.set_shape([None])
    labels.set_shape([None])
    label_length.set_shape([])
    return signal, labels, label_length

  def __create_dataset(self, entries, text_featurizer, sample_rate, sort,
                       augmentations, batch_size, repeat=1, shuffle=True, preemph=None):
    def _gen_data():
      for audio_file, au, transcript in entries:
        yield audio_file, au, transcript

    dataset = tf.data.Dataset.from_generator(_gen_data,
                                             output_types=(tf.string, tf.int32, tf.string))
    dataset = dataset.map(functools.partial(self.preprocess_map_fn,
                                            text_featurizer=text_featurizer,
                                            augmentations=augmentations,
                                            sample_rate=sample_rate, coeff=preemph),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # # Padding the features to its max length dimensions
    dataset = dataset.repeat(repeat)
    if shuffle and not sort:
      dataset = dataset.shuffle(self.num_cpus,
                                reshuffle_each_iteration=True)  # shuffle elements in batches
    dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes=(
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([])
      ),
      padding_values=(
        0.,
        text_featurizer.num_classes - 1,
        0
      )
    )
    if shuffle and sort:
      dataset = dataset.shuffle(self.num_cpus,
                                reshuffle_each_iteration=True)  # shuffle the sorted batches
    # Prefetch to improve speed of input length
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
