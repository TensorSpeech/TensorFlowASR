from __future__ import absolute_import

import os
import sys
import functools
import glob
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


def to_tfrecord(audio, au, transcript):
  feature = {
    "audio":      _bytestring_feature([audio]),
    "au":         _int64_feature([au]),
    "transcript": _bytestring_feature([transcript])
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
    self.create_tfrecords(augmentations, sortagrad)
    if self.mode == "train":
      return self.get_dataset_from_tfrecords(text_featurizer, augmentations=augmentations,
                                             sample_rate=sample_rate, preemph=preemph,
                                             batch_size=batch_size, repeat=repeat,
                                             sort=sortagrad, shuffle=True)
    elif self.mode in ["eval", "test"]:
      return self.get_dataset_from_tfrecords(text_featurizer, augmentations=augmentations,
                                             sample_rate=sample_rate, preemph=preemph,
                                             batch_size=batch_size, repeat=repeat,
                                             sort=sortagrad, shuffle=False)
    else:
      raise ValueError(f"Mode must be either 'train', 'eval' or 'test': {self.mode}")

  @staticmethod
  def write_tfrecord_file(splitted_entries):
    shard_path, entries = splitted_entries
    with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
      for audio_file, au, transcript in entries:
        with open(audio_file, "rb") as f:
          audio = f.read()
        example = to_tfrecord(audio, int(au), bytes(transcript, "utf-8"))
        out.write(example.SerializeToString())
        sys.stdout.write("\033[K")
        print(f"Processed: {audio_file}", end="\r")
    print(f"\nCreated {shard_path}")

  def create_tfrecords(self, augmentations=tuple([None]), sortagrad=False):
    print(f"Creating {self.mode}.tfrecord ...")
    if glob.glob(os.path.join(self.tfrecord_dir, f"{self.mode}*.tfrecord")):
      return
    entries = self.create_entries(augmentations, sortagrad)

    def get_shard_path(shard_id):
      return os.path.join(self.tfrecord_dir, f"{self.mode}_{shard_id}.tfrecord")

    shards = [get_shard_path(idx) for idx in range(1, self.num_cpus + 1)]

    splitted_entries = np.array_split(entries, self.num_cpus)
    with multiprocessing.Pool(self.num_cpus) as pool:
      pool.map(self.write_tfrecord_file, zip(shards, splitted_entries))

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

  @staticmethod
  def preprocess(audio, au, transcript, text_featurizer, augmentations, sample_rate, preemph):
    signal = read_raw_audio(audio.numpy(), sample_rate)
    if augmentations[int(au)] is not None:
      signal = augmentations[int(au)](signal=signal, sample_rate=sample_rate)
    signal = preemphasis(signal, coeff=preemph)
    label = text_featurizer.compute_label_features(transcript.numpy().decode("utf-8"))
    label_length = tf.cast(tf.shape(label)[0], tf.int32)
    return tf.convert_to_tensor(signal, tf.float32), label, label_length

  def get_dataset_from_tfrecords(self, text_featurizer, augmentations, sample_rate, preemph,
                                 batch_size, repeat=1, sort=False, shuffle=True):

    def parse(record):
      feature_description = {
        "audio":      tf.io.FixedLenFeature([], tf.string),
        "au":         tf.io.FixedLenFeature([], tf.int64),
        "transcript": tf.io.FixedLenFeature([], tf.string)
      }
      example = tf.io.parse_single_example(record, feature_description)
      signal, label, label_length = tf.py_function(
        functools.partial(self.preprocess, text_featurizer=text_featurizer,
                          augmentations=augmentations, sample_rate=sample_rate, preemph=preemph),
        inp=[example["audio"], example["au"], example["transcript"]],
        Tout=(tf.float32, tf.int32, tf.int32))
      return signal, label, label_length

    pattern = os.path.join(self.tfrecord_dir, f"{self.mode}*.tfrecord")
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)
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
