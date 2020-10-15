# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import glob
import multiprocessing
import os

import numpy as np
import tensorflow as tf

from .base_dataset import BaseDataset
from ..featurizers.speech_featurizers import read_raw_audio, SpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils.utils import bytestring_feature, print_one_line, get_num_batches

AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_SHARDS = 16


def to_tfrecord(path, audio, transcript):
    feature = {
        "path": bytestring_feature([path]),
        "audio": bytestring_feature([audio]),
        "transcript": bytestring_feature([transcript])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord_file(splitted_entries):
    shard_path, entries = splitted_entries
    with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
        for audio_file, _, transcript in entries:
            with open(audio_file, "rb") as f:
                audio = f.read()
            example = to_tfrecord(bytes(audio_file, "utf-8"), audio, bytes(transcript, "utf-8"))
            out.write(example.SerializeToString())
            print_one_line("Processed:", audio_file)
    print(f"\nCreated {shard_path}")


class ASRDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: dict = None,
                 cache: bool = False,
                 shuffle: bool = False):
        super(ASRDataset, self).__init__(data_paths, augmentations, cache, shuffle, stage)
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def read_entries(self):
        lines = []
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                lines += temp_lines[1:]
        # The files is "\t" seperated
        lines = [line.split("\t", 2) for line in lines]
        lines = np.array(lines)
        if self.shuffle:
            np.random.shuffle(lines)  # Mix transcripts.tsv
        self.total_steps = len(lines)
        return lines

    def preprocess(self, audio, transcript):
        with tf.device("/CPU:0"):
            signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)

            signal = self.augmentations["before"].augment(signal)

            features = self.speech_featurizer.extract(signal)

            features = self.augmentations["after"].augment(features)

            label = self.text_featurizer.extract(transcript.decode("utf-8"))
            label_length = tf.cast(tf.shape(label)[0], tf.int32)
            pred_inp = self.text_featurizer.prepand_blank(label)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            return features, input_length, label, label_length, pred_inp

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(TFRECORD_SHARDS, reshuffle_each_iteration=True)

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([]),
                tf.TensorShape([None])
            ),
            padding_values=("", 0., 0, self.text_featurizer.blank,
                            0, self.text_featurizer.blank),
            drop_remainder=True
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size)
        return dataset

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, batch_size):
        raise NotImplementedError()


class ASRTFRecordDataset(ASRDataset):
    """ Dataset for ASR using TFRecords """

    def __init__(self,
                 data_paths: list,
                 tfrecords_dir: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 stage: str,
                 augmentations: dict = None,
                 cache: bool = False,
                 shuffle: bool = False):
        super(ASRTFRecordDataset, self).__init__(
            stage, speech_featurizer, text_featurizer,
            data_paths, augmentations, cache, shuffle
        )
        self.tfrecords_dir = tfrecords_dir
        if not os.path.exists(self.tfrecords_dir):
            os.makedirs(self.tfrecords_dir)

    def create_tfrecords(self):
        if not os.path.exists(self.tfrecords_dir):
            os.makedirs(self.tfrecords_dir)

        entries = self.read_entries()
        if len(entries) <= 0:
            return False

        if glob.glob(os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")):
            print(f"TFRecords're already existed: {self.stage}")
            return True

        print(f"Creating {self.stage}.tfrecord ...")

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, TFRECORD_SHARDS + 1)]

        splitted_entries = np.array_split(entries, TFRECORD_SHARDS)
        with multiprocessing.Pool(TFRECORD_SHARDS) as pool:
            pool.map(write_tfrecord_file, zip(shards, splitted_entries))

        return True

    @tf.function
    def parse(self, record):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "transcript": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)

        features, input_length, label, label_length, pred_inp = tf.numpy_function(
            self.preprocess,
            inp=[example["audio"], example["transcript"]],
            Tout=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        )
        return example["path"], features, input_length, label, label_length, pred_inp

    def create(self, batch_size):
        # Create TFRecords dataset
        have_data = self.create_tfrecords()
        if not have_data:
            return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(
            files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)

        return self.process(dataset, batch_size)


class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """

    def preprocess(self, path, transcript):
        data = super(ASRSliceDataset, self).preprocess(path.decode("utf-8"), transcript)
        return (path, *data)

    @tf.function
    def parse(self, record):
        return tf.numpy_function(
            self.preprocess,
            inp=[record[0], record[1]],
            Tout=[tf.string, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32]
        )

    def create(self, batch_size):
        entries = self.read_entries()
        if len(entries) == 0:
            return None
        entries = np.delete(entries, 1, 1)  # Remove unused duration

        dataset = tf.data.Dataset.from_tensor_slices(entries)

        return self.process(dataset, batch_size)
