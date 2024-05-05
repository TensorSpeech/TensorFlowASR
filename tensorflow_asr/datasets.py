# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

# Dataset Structures :kissing:

# To make a custom dataset, inherit the `BaseDataset` class and override following methods:

# 1. `create` to create `tf.data.Dataset` instance.
# 2. `parse` for transforming `tf.data.Dataset` during creation by applyting `tf.data.Dataset.map` function.

# _Note_: To create transcripts for **librispeech**, see [create_librispeech_trans.py](../../scripts/create_librispeech_trans.py)

# ## ASR Datasets

# An ASR dataset is some `.tsv` files in format: `PATH\tDURATION\tTRANSCRIPT`. You must create those files by your own with your own data and methods.

# **Note**: Each `.tsv` file must include a header `PATH\tDURATION\tTRANSCRIPT`
# because it will remove these headers when loading dataset, otherwise you will lose 1 data file :sob:

# **For transcript**, if you want to include characters such as dots, commas, double quote, etc.. you must create your own `.txt` vocabulary file.
# Default is [English](../featurizers/english.txt)

# **Inputs**

# ```python
# class ASRTFRecordDataset(ASRDataset):
#     """ Dataset for ASR using TFRecords """

# class ASRSliceDataset(ASRDataset):
#     """ Dataset for ASR using Slice """
# ```

# **Outputs when iterating dataset**

# ```python
# (
#     {
#         "inputs": ...,
#         "inputs_length": ...,
#         "predictions": ...,
#         "predictions_length": ...,
#     },
#     {
#         "labels": ...,
#         "labels_length": ...
#     }
# )
# ```

# Where `predictions` and `predictions_length` are the label prepanded by blank and its length for training *Transducer*

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import tensorflow as tf
import tqdm

from tensorflow_asr import schemas
from tensorflow_asr.configs import Config, DatasetConfig
from tensorflow_asr.tokenizers import Tokenizer
from tensorflow_asr.utils import data_util, feature_util, file_util, math_util

logger = tf.get_logger()


@dataclass
class ASR_DATASER_TYPES:
    TFRECORD: str = "tfrecord"
    SLICE: str = "slice"
    GENERATOR: str = "generator"


def get(
    tokenizer: Tokenizer,
    dataset_config: DatasetConfig,
    dataset_type: str,
):
    if dataset_type == ASR_DATASER_TYPES.TFRECORD:
        return ASRTFRecordDataset(tokenizer=tokenizer, **vars(dataset_config))
    if dataset_type == ASR_DATASER_TYPES.SLICE:
        return ASRSliceDataset(tokenizer=tokenizer, **vars(dataset_config))
    if dataset_type == ASR_DATASER_TYPES.GENERATOR:
        return ASRDataset(tokenizer=tokenizer, **vars(dataset_config))
    raise ValueError(f"dataset_type must in {asdict(ASR_DATASER_TYPES()).values()}")


def get_global_shape(
    config: Config,
    strategy,
    *datasets,
    batch_size: int = None,
    ga_steps: int = None,
):
    batch_size = (batch_size or config.learning_config.running_config.batch_size) * strategy.num_replicas_in_sync
    ds_batch_size = batch_size
    if ga_steps is not None and ga_steps > 1:
        ds_batch_size *= ga_steps

    max_input_length, max_label_length = 0, 0
    for dset in datasets:
        max_input_length = max(max_input_length, dset.max_input_length or 0)
        max_label_length = max(max_label_length, dset.max_label_length or 0)
    max_input_length = None if max_input_length == 0 else max_input_length
    max_label_length = None if max_label_length == 0 else max_label_length

    input_shape = [max_input_length]
    prediction_shape = [max_label_length + 1] if max_label_length else [None]
    label_shape = [max_label_length]
    padded_shapes = (
        schemas.TrainInput(
            inputs=tf.TensorShape(input_shape),
            inputs_length=tf.TensorShape([]),
            predictions=tf.TensorShape(prediction_shape),
            predictions_length=tf.TensorShape([]),
        ),
        schemas.TrainLabel(
            labels=tf.TensorShape(label_shape),
            labels_length=tf.TensorShape([]),
        ),
    )

    return dict(
        ds_batch_size=ds_batch_size,
        batch_size=batch_size,
        input_shape=input_shape,
        prediction_shape=prediction_shape,
        label_shape=label_shape,
        padded_shapes=padded_shapes,
    )


BUFFER_SIZE = 100
TFRECORD_BUFFER_SIZE = 32 * 1024 * 1024
TFRECORD_SHARDS = 16
AUTOTUNE = int(os.environ.get("AUTOTUNE") or tf.data.AUTOTUNE)


class BaseDataset:
    """Based dataset for all models"""

    def __init__(
        self,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        sample_rate: int = 16000,
        stage: str = "train",
        name: str = "",
        **kwargs,
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.cache = cache  # whether to cache transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        self.buffer_size = buffer_size  # shuffle buffer size
        self.stage = stage  # for defining tfrecords files
        self.enabled = enabled
        self.drop_remainder = drop_remainder  # whether to drop remainder for multi gpu training
        self.indefinite = indefinite  # Whether to make dataset repeat indefinitely -> avoid the potential last partial batch
        self.total_steps = None  # for better training visualization
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.name = name

    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    def create(self, batch_size):
        raise NotImplementedError()


class ASRDataset(BaseDataset):
    """Dataset for ASR using Generator"""

    def __init__(
        self,
        stage: str,
        tokenizer: Tokenizer,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        indefinite: bool = True,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        buffer_size: int = BUFFER_SIZE,
        sample_rate: int = 16000,
        name: str = "",
        **kwargs,
    ):
        super().__init__(
            data_paths=data_paths,
            cache=cache,
            shuffle=shuffle,
            stage=stage,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
            enabled=enabled,
            metadata=metadata,
            indefinite=indefinite,
            sample_rate=sample_rate,
            name=name,
            **kwargs,
        )
        self.entries = []
        self.tokenizer = tokenizer
        self.max_input_length = None
        self.max_label_length = None
        self.load_metadata()

    # -------------------------------- metadata -------------------------------------

    def compute_metadata(self):
        self.max_input_length = 0 if self.max_input_length is None else self.max_input_length
        self.max_label_length = 0 if self.max_label_length is None else self.max_label_length
        if self.max_input_length > 0 and self.max_label_length > 0:
            return  # already computed
        self.read_entries()
        for _, duration, transcript in tqdm.tqdm(self.entries, desc=f"Computing metadata for entries in {self.stage} dataset"):
            input_length = math_util.get_nsamples(duration, self.sample_rate)
            label = self.tokenizer.tokenize(transcript).numpy()
            label_length = len(label)
            self.max_input_length = max(self.max_input_length, input_length)
            self.max_label_length = max(self.max_label_length, label_length)

    def save_metadata(self):
        if self.metadata is None:
            return
        self.metadata = file_util.preprocess_paths(self.metadata)
        if tf.io.gfile.exists(self.metadata):
            with tf.io.gfile.GFile(self.metadata, "r") as f:
                try:
                    content = json.loads(f.read())
                except json.JSONDecodeError as e:
                    raise ValueError(f"File {self.metadata} is currently not in json format. Please update the file") from e
        else:
            content = {}
        content[self.stage] = dict(
            max_input_length=self.max_input_length,
            max_label_length=self.max_label_length,
            num_entries=self.total_steps,
        )
        with tf.io.gfile.GFile(self.metadata, "w") as f:
            f.write(json.dumps(content, indent=2))
        logger.info(f"Metadata written to {self.metadata}")

    def load_metadata(self):
        if self.metadata is None:
            return
        if not self.enabled:
            return
        content = None
        self.metadata = file_util.preprocess_paths(self.metadata)
        if tf.io.gfile.exists(self.metadata):
            logger.info(f"Loading metadata from {self.metadata} ...")
            with tf.io.gfile.GFile(self.metadata, "r") as f:
                try:
                    content = json.loads(f.read()).get(self.stage, {})
                except json.JSONDecodeError as e:
                    raise ValueError(f"File {self.metadata} must be in json format") from e
        if not content:
            return
        self.max_input_length = content.get("max_input_length")
        self.max_label_length = content.get("max_label_length")
        self.total_steps = int(content.get("num_entries", 0))
        self.num_entries = self.total_steps

    def update_metadata(self):
        self.load_metadata()
        self.compute_metadata()
        self.save_metadata()

    # -------------------------------- ENTRIES -------------------------------------

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.data_paths = file_util.preprocess_paths(self.data_paths, enabled=self.enabled, check_exists=True)
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                for line in f.read().splitlines()[1:]:  # Skip the header of tsv file
                    self.entries.append(line.split("\t", 2))  # The files is "\t" seperated
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)
        self.num_entries = self.total_steps

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def generator(self):
        for path, _, transcript in self.entries:
            audio = data_util.load_and_convert_to_wav(path, sample_rate=self.sample_rate).numpy()
            yield bytes(path, "utf-8"), audio, bytes(transcript, "utf-8")

    def _process_item(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        inputs = data_util.read_raw_audio(audio)
        inputs_length = tf.shape(inputs)[0]

        labels = self.tokenizer.tokenize(transcript)
        labels_length = tf.shape(labels, out_type=tf.int32)[0]

        predictions = self.tokenizer.prepand_blank(labels)
        predictions_length = tf.shape(predictions, out_type=tf.int32)[0]

        return path, inputs, inputs_length, labels, labels_length, predictions, predictions_length

    def parse(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        (
            _,
            inputs,
            inputs_length,
            labels,
            labels_length,
            predictions,
            predictions_length,
        ) = self._process_item(path=path, audio=audio, transcript=transcript)
        return (
            schemas.TrainInput(inputs=inputs, inputs_length=inputs_length, predictions=predictions, predictions_length=predictions_length),
            schemas.TrainLabel(labels=labels, labels_length=labels_length),
        )

    # -------------------------------- CREATION -------------------------------------

    def process(self, dataset: tf.data.Dataset, batch_size: int, padded_shapes=None):
        if self.cache:
            dataset = dataset.cache()  # cache original (unchanged data)

        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE, deterministic=False)
        self.total_steps = math_util.get_num_batches(self.num_entries, batch_size, drop_remainders=self.drop_remainder)

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size or self.num_entries, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        if padded_shapes is None:
            padded_shapes = (
                schemas.TrainInput(
                    inputs=tf.TensorShape([self.max_input_length]),
                    inputs_length=tf.TensorShape([]),
                    predictions=tf.TensorShape([self.max_label_length + 1 if self.max_label_length else None]),
                    predictions_length=tf.TensorShape([]),
                ),
                schemas.TrainLabel(
                    labels=tf.TensorShape([self.max_label_length]),
                    labels_length=tf.TensorShape([]),
                ),
            )

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            padding_values=(
                schemas.TrainInput(inputs=0.0, inputs_length=0, predictions=self.tokenizer.blank, predictions_length=0),
                schemas.TrainLabel(labels=self.tokenizer.blank, labels_length=0),
            ),
            drop_remainder=self.drop_remainder,
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
        )
        return self.process(dataset, batch_size, padded_shapes=padded_shapes)


class ASRTFRecordDataset(ASRDataset):
    """Dataset for ASR using TFRecords"""

    def __init__(
        self,
        data_paths: list,
        tfrecords_dir: str,
        tokenizer: Tokenizer,
        stage: str,
        tfrecords_shards: int = TFRECORD_SHARDS,
        cache: bool = False,
        shuffle: bool = False,
        enabled: bool = True,
        metadata: str = None,
        indefinite: bool = True,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        tfrecords_buffer_size: int = TFRECORD_BUFFER_SIZE,
        compression_type: str = "GZIP",
        sample_rate: int = 16000,
        name: str = "",
        **kwargs,
    ):
        super().__init__(
            stage=stage,
            tokenizer=tokenizer,
            data_paths=data_paths,
            cache=cache,
            shuffle=shuffle,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
            enabled=enabled,
            metadata=metadata,
            indefinite=indefinite,
            sample_rate=sample_rate,
            name=name,
            **kwargs,
        )
        if not self.stage:
            raise ValueError("stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0:
            raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        self.tfrecords_buffer_size = tfrecords_buffer_size
        self.compression_type = compression_type

    def write_tfrecord_file(self, splitted_entries: tuple):
        shard_path, entries = splitted_entries
        logger.info(f"Processing {shard_path} ...")
        with tf.io.TFRecordWriter(shard_path, options=tf.io.TFRecordOptions(compression_type=self.compression_type)) as writer:
            for path, _, transcript in entries:
                audio = data_util.load_and_convert_to_wav(path, sample_rate=self.sample_rate).numpy()
                feature = dict(
                    path=feature_util.bytestring_feature([path.encode("utf-8")]),
                    audio=feature_util.bytestring_feature([audio]),
                    transcript=feature_util.bytestring_feature([transcript.encode("utf-8")]),
                )
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        logger.info(f"Created {shard_path}")

    def create_tfrecords(self):
        if not self.tfrecords_dir:
            return False
        self.tfrecords_dir = file_util.preprocess_paths(self.tfrecords_dir, isdir=True, enabled=self.enabled)

        if tf.io.gfile.glob(os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")):
            logger.info(f"TFRecords're already existed: {self.stage}")
            return True

        logger.info(f"Creating {self.stage}.tfrecord ...")

        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return False

        def get_shard_path(shard_id: int):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(self.entries, self.tfrecords_shards)
        for entries in zip(shards, splitted_entries):
            self.write_tfrecord_file(entries)

        return True

    def parse(self, record: tf.Tensor, **kwargs):
        feature_description = dict(
            path=tf.io.FixedLenFeature([], tf.string),
            audio=tf.io.FixedLenFeature([], tf.string),
            transcript=tf.io.FixedLenFeature([], tf.string),
        )
        example = tf.io.parse_single_example(record, feature_description)
        return super().parse(**example)

    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        have_data = self.create_tfrecords()
        if not have_data:
            return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern, shuffle=self.shuffle)
        ignore_order = tf.data.Options()
        ignore_order.deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(
            files_ds, compression_type=self.compression_type, buffer_size=self.tfrecords_buffer_size, num_parallel_reads=AUTOTUNE
        )

        return self.process(dataset, batch_size, padded_shapes=padded_shapes)


class ASRSliceDataset(ASRDataset):
    """Dataset for ASR using Slice"""

    def load(self, record):
        audio = tf.numpy_function(
            lambda path: data_util.load_and_convert_to_wav(path.decode("utf-8"), sample_rate=self.sample_rate).numpy(),
            inp=[record[0]],
            Tout=tf.string,
        )
        return record[0], audio, record[2]

    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None

        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        options = tf.data.Options()
        options.deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE, deterministic=False)

        return self.process(dataset, batch_size, padded_shapes=padded_shapes)
