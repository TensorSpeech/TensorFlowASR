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

import os
import json
import tqdm
import numpy as np
import tensorflow as tf

from ..augmentations.augments import Augmentation
from .base_dataset import BaseDataset, BUFFER_SIZE, TFRECORD_SHARDS, AUTOTUNE
from ..featurizers.speech_featurizers import load_and_convert_to_wav, read_raw_audio, tf_read_raw_audio, SpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils.utils import bytestring_feature, get_num_batches, preprocess_paths


class ASRDataset(BaseDataset):
    """ Dataset for ASR using Generator """

    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 use_tf: bool = False,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        super(ASRDataset, self).__init__(
            data_paths=data_paths, augmentations=augmentations,
            cache=cache, shuffle=shuffle, stage=stage, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    # -------------------------------- metadata -------------------------------------

    def compute_metadata(self):
        self.read_entries()
        for _, duration, indices in tqdm.tqdm(self.entries, desc=f"Computing metadata for entries in {self.stage} dataset"):
            input_length = self.speech_featurizer.get_length_from_duration(duration)
            label = str(indices).split()
            label_length = len(label)
            self.speech_featurizer.update_length(input_length)
            self.text_featurizer.update_length(label_length)

    def save_metadata(self, metadata_prefix: str = None):
        if metadata_prefix is None: return
        metadata_path = preprocess_paths(metadata_prefix) + ".metadata.json"
        if tf.io.gfile.exists(metadata_path):
            with tf.io.gfile.GFile(metadata_path, "r") as f:
                content = json.loads(f.read())
        else:
            content = {}
        content[self.stage] = {
            "max_input_length": self.speech_featurizer.max_length,
            "max_label_length": self.text_featurizer.max_length,
            "num_entries": self.total_steps
        }
        with tf.io.gfile.GFile(metadata_path, "w") as f:
            f.write(json.dumps(content, indent=2))
        print(f"metadata written to {metadata_path}")

    def load_metadata(self, metadata_prefix: str = None):
        if metadata_prefix is None: return
        metadata_path = preprocess_paths(metadata_prefix) + ".metadata.json"
        if tf.io.gfile.exists(metadata_path):
            print(f"Loading metadata from {metadata_path} ...")
            with tf.io.gfile.GFile(metadata_path, "r") as f:
                content = json.loads(f.read()).get(self.stage, {})
                self.speech_featurizer.update_length(int(content.get("max_input_length", 0)))
                self.text_featurizer.update_length(int(content.get("max_label_length", 0)))
                self.total_steps = int(content.get("num_entries", 0))

    def update_metadata(self, metadata_prefix: str = None):
        self.load_metadata(metadata_prefix)
        self.compute_metadata()
        self.save_metadata(metadata_prefix)

    # -------------------------------- ENTRIES -------------------------------------

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0: return
        self.entries = []
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                self.entries += temp_lines[1:]
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
        self.entries = np.array(self.entries)
        if self.shuffle: np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def generator(self):
        for path, _, indices in self.entries:
            audio = load_and_convert_to_wav(path).numpy()
            yield bytes(path, "utf-8"), audio, bytes(indices, "utf-8")

    def preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            def fn(_path: bytes, _audio: bytes, _indices: bytes):
                signal = read_raw_audio(_audio, sample_rate=self.speech_featurizer.sample_rate)

                signal = self.augmentations.before.augment(signal)

                features = self.speech_featurizer.extract(signal)

                features = self.augmentations.after.augment(features)

                label = tf.strings.to_number(tf.strings.split(_indices), out_type=tf.int32)
                label_length = tf.cast(tf.shape(label)[0], tf.int32)
                prediction = self.text_featurizer.prepand_blank(label)
                prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)
                features = tf.convert_to_tensor(features, tf.float32)
                input_length = tf.cast(tf.shape(features)[0], tf.int32)

                return _path, features, input_length, label, label_length, prediction, prediction_length

            return tf.numpy_function(
                fn, inp=[path, audio, indices],
                Tout=[tf.string, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
            )

    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)

            signal = self.augmentations.before.augment(signal)

            features = self.speech_featurizer.tf_extract(signal)

            features = self.augmentations.after.augment(features)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)
            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            return path, features, input_length, label, label_length, prediction, prediction_length

    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        if self.use_tf: return self.tf_preprocess(path, audio, indices)
        return self.preprocess(path, audio, indices)

    # -------------------------------- CREATION -------------------------------------

    def process(self, dataset: tf.data.Dataset, batch_size: int):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite:
            dataset = dataset.repeat()

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape(self.text_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape(self.text_featurizer.prepand_shape),
                tf.TensorShape([]),
            ),
            padding_values=(None, 0., 0, self.text_featurizer.blank, 0, self.text_featurizer.blank, 0),
            drop_remainder=self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder)
        return dataset

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        )
        return self.process(dataset, batch_size)


class ASRTFRecordDataset(ASRDataset):
    """ Dataset for ASR using TFRecords """

    def __init__(self,
                 data_paths: list,
                 tfrecords_dir: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 stage: str,
                 augmentations: Augmentation = Augmentation(None),
                 tfrecords_shards: int = TFRECORD_SHARDS,
                 cache: bool = False,
                 shuffle: bool = False,
                 use_tf: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        super(ASRTFRecordDataset, self).__init__(
            stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite
        )
        if not self.stage: raise ValueError("stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0: raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        if not tf.io.gfile.exists(self.tfrecords_dir): tf.io.gfile.makedirs(self.tfrecords_dir)

    @staticmethod
    def write_tfrecord_file(splitted_entries):
        shard_path, entries = splitted_entries

        def parse(record):
            def fn(path, indices):
                audio = load_and_convert_to_wav(path.decode("utf-8")).numpy()
                feature = {
                    "path": bytestring_feature([path]),
                    "audio": bytestring_feature([audio]),
                    "indices": bytestring_feature([indices])
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                return example.SerializeToString()
            return tf.numpy_function(fn, inp=[record[0], record[2]], Tout=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices(entries)
        dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)
        writer = tf.data.experimental.TFRecordWriter(shard_path, compression_type="ZLIB")
        print(f"Processing {shard_path} ...")
        writer.write(dataset)
        print(f"Created {shard_path}")

    def create_tfrecords(self):
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

        if tf.io.gfile.glob(os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")):
            print(f"TFRecords're already existed: {self.stage}")
            return True

        print(f"Creating {self.stage}.tfrecord ...")

        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return False

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(self.entries, self.tfrecords_shards)
        for entries in zip(shards, splitted_entries):
            self.write_tfrecord_file(entries)

        return True

    def parse(self, record: tf.Tensor):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        if self.use_tf: return self.tf_preprocess(**example)
        return self.preprocess(**example)

    def create(self, batch_size: int):
        have_data = self.create_tfrecords()
        if not have_data: return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type="ZLIB", num_parallel_reads=AUTOTUNE)

        return self.process(dataset, batch_size)


class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes): return load_and_convert_to_wav(path.decode("utf-8")).numpy()
        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        return record[0], audio, record[2]

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)


class ASRMaskedSliceDataset(ASRSliceDataset):
    """ Dataset for ASR with rolling mask """

    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 use_tf: bool = False,
                 buffer_size: int = BUFFER_SIZE,
                 history_window_size: int = 3,
                 input_chunk_duration: int = 250,
                 **kwargs):
        super(ASRMaskedSliceDataset, self).__init__(
            data_paths=data_paths, augmentations=augmentations,
            cache=cache, shuffle=shuffle, stage=stage, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite,
            speech_featurizer=speech_featurizer, text_featurizer=text_featurizer
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.history_window_size = history_window_size
        self.input_chunk_size = input_chunk_duration * self.speech_featurizer.sample_rate // 1000

    def calculate_mask(self, num_frames):
        frame_step = self.speech_featurizer.frame_step
        frames_per_chunk = self.input_chunk_size // frame_step

        time_reduction_factor = 4 # TODO: Get time_reduction_factor from config or model.
        num_frames = tf.cast(tf.math.ceil(num_frames / time_reduction_factor), tf.int32)

        def _calculate_mask(num_frames, frames_per_chunk, history_window_size):
            mask = np.zeros((num_frames, num_frames), dtype=np.int32)
            for i in range(num_frames):
                # Frames in the same chunk can see each other
                # If frames in `history_window_size` are in other chunks, the full chunks are visible
                current_chunk_index = i // frames_per_chunk
                history_chunk_index = (i - history_window_size) // frames_per_chunk
                for curr in range(history_chunk_index, current_chunk_index + 1):
                    for j in range(frames_per_chunk):
                        base_index = curr * frames_per_chunk
                        if base_index + j < 0 or base_index + j >= num_frames:
                            continue
                        mask[i, base_index + j] = 1
            return mask

        return tf.numpy_function(
            _calculate_mask, inp=[num_frames, frames_per_chunk, self.history_window_size], Tout=tf.int32
        )

    def preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        preprocessed_inputs = super(ASRMaskedSliceDataset, self).preprocess(path, audio, indices)

        input_length = preprocessed_inputs[2]
        mask = self.calculate_mask(input_length)
        mask.set_shape((None, None))

        return (*preprocessed_inputs, mask)

    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        preprocessed_inputs = super(ASRMaskedSliceDataset, self).tf_preprocess(path, audio, indices)

        input_length = preprocessed_inputs[2]
        mask = self.calculate_mask(input_length)
        mask.set_shape((None, None))

        return (*preprocessed_inputs, mask)

    # -------------------------------- CREATION -------------------------------------

    def process(self, dataset: tf.data.Dataset, batch_size: int):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite:
            dataset = dataset.repeat()

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape(self.text_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape(self.text_featurizer.prepand_shape),
                tf.TensorShape([]),
                tf.TensorShape([self.speech_featurizer.shape[0], self.speech_featurizer.shape[0]])
            ),
            padding_values=(None, 0., 0, self.text_featurizer.blank, 0, self.text_featurizer.blank, 0, 0),
            drop_remainder=self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder)
        return dataset

# TODO: Create masked TFRecords dataset
