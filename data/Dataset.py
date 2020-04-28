from __future__ import absolute_import

import os
import sys
import functools
import multiprocessing
import numpy as np
import tensorflow as tf
from featurizers.SpeechFeaturizer import read_raw_audio, preemphasis

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _float_feature(list_of_floats):
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(list_of_ints):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def to_tfrecord(audio, labels, labels_length):
  feature = {
    "signal":       _float_feature(audio),
    "label":        _int64_feature(labels),
    "label_length": _int64_feature([labels_length])
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


class Dataset:
  def __init__(self, data_path, tfrecords_dir, mode="train"):
    self.data_path = data_path
    self.tfrecord_dir = tfrecords_dir
    self.mode = mode
    self.num_cpus = multiprocessing.cpu_count()

  def __call__(self, text_featurizer, sample_rate=16000, preemph=0.95, batch_size=32,
               repeat=1, augmentations=tuple([None]), sortagrad=False):
    self.create_tfrecords(text_featurizer, augmentations, sample_rate, preemph, sortagrad)
    if self.mode == "train":
      return self.get_dataset_from_tfrecords(text_featurizer, batch_size, repeat=repeat,
                                             sort=sortagrad, shuffle=True)
    elif self.mode in ["eval", "test"]:
      return self.get_dataset_from_tfrecords(text_featurizer, batch_size, repeat=repeat,
                                             sort=sortagrad, shuffle=False)
    else:
      raise ValueError(f"Mode must be either 'train', 'eval' or 'test': {self.mode}")

  @staticmethod
  def write_tfrecord_file(splitted_entries, text_featurizer, augmentations, sample_rate, preemph):
    shard_path, entries = splitted_entries
    if os.path.exists(shard_path):
      return
    with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
      for audio_file, au, transcript in entries:
        signal = read_raw_audio(audio_file, sample_rate)
        if augmentations[int(au)] is not None:
          signal = augmentations[int(au)](signal=signal, sample_rate=sample_rate)
        signal = preemphasis(signal, coeff=preemph)
        label = text_featurizer.compute_label_features(transcript)
        label_length = tf.cast(tf.shape(label)[0], tf.int64)

        signal = tf.expand_dims(signal, - 1)
        label = tf.expand_dims(label, - 1)

        example = to_tfrecord(signal, label, label_length)
        out.write(example.SerializeToString())
        sys.stdout.write("\033[K")
        print(f"Processed: {audio_file}", end="\r")
    print(f"\nCreated {shard_path}")

  def create_tfrecords(self, text_featurizer, augmentations=tuple([None]),
                       sample_rate=16000, preemph=0.95, sortagrad=False):
    print(f"Creating {self.mode}.tfrecord ...")
    entries = self.create_entries(augmentations, sortagrad)

    def get_shard_path(shard_id):
      return os.path.join(self.tfrecord_dir, f"{self.mode}_{shard_id}.tfrecord")

    shards = [get_shard_path(idx) for idx in range(1, self.num_cpus + 1)]

    splitted_entries = np.array_split(entries, self.num_cpus)
    with multiprocessing.Pool(self.num_cpus) as pool:
      pool.map(functools.partial(self.write_tfrecord_file, text_featurizer=text_featurizer,
                                 augmentations=augmentations, sample_rate=sample_rate, preemph=preemph),
               zip(shards, splitted_entries))

  @staticmethod
  def entries_map_fn(splitted_lines, augmentations):
    results = []
    for path, _, transcript in splitted_lines:
      for idx, au in enumerate(augmentations):
        results.append([path, idx, transcript])
    return np.array(results)

  def create_entries(self, augmentations, sort=False):
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

  def get_dataset_from_tfrecords(self, text_featurizer, batch_size, repeat=1, sort=False, shuffle=True):

    def parse(record):
      feature_description = {
        "signal":       tf.io.FixedLenSequenceFeature(1, tf.float32),
        "label":        tf.io.FixedLenSequenceFeature(1, tf.int64),
        "label_length": tf.io.FixedLenFeature([], tf.int64)
      }
      example = tf.io.parse_single_example(record, feature_description)
      return (tf.squeeze(example["signal"], -1),
              tf.squeeze(tf.cast(example["label"], tf.int32), - 1),
              tf.cast(example["label_length"], tf.int32))

    pattern = os.path.join(self.tfrecord_dir, f"{self.mode}*.tfrecord")
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)
    print(dataset.element_spec)
    # # Padding the features to its max length dimensions
    dataset = dataset.repeat(repeat)
    if shuffle and not sort:
      dataset = dataset.shuffle(100, reshuffle_each_iteration=True)  # shuffle elements in batches
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
      dataset = dataset.shuffle(100, reshuffle_each_iteration=True)  # shuffle the sorted batches
    # Prefetch to improve speed of input length
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
