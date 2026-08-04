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

import gzip
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

from tensorflow_asr import schemas, tf
from tensorflow_asr.abstracts import AbstractDataset, AbstractTokenizer
from tensorflow_asr.configs import Config, DatasetConfig
from tensorflow_asr.utils import data_util, feature_util, file_util, math_util

logger = logging.getLogger(__name__)


@dataclass
class ASR_DATASER_TYPES:
    TFRECORD: str = "tfrecord"
    SLICE: str = "slice"
    GENERATOR: str = "generator"
    HUGGINGFACE: str = "huggingface"


def get(
    tokenizer: AbstractTokenizer,
    dataset_config: DatasetConfig,
    dataset_type: str,
    dataset_cache: bool = False,
):
    dataset_config.cache = dataset_cache
    if dataset_type == ASR_DATASER_TYPES.TFRECORD:
        return ASRTFRecordDataset(tokenizer=tokenizer, **vars(dataset_config))
    if dataset_type == ASR_DATASER_TYPES.SLICE:
        return ASRSliceDataset(tokenizer=tokenizer, **vars(dataset_config))
    if dataset_type == ASR_DATASER_TYPES.GENERATOR:
        return ASRDataset(tokenizer=tokenizer, **vars(dataset_config))
    raise ValueError(f"dataset_type must in {asdict(ASR_DATASER_TYPES()).values()}")


def get_global_shape(
    config: Config,
    strategy: tf.distribute.Strategy,
    *datasets: "ASRDataset",
    batch_size: int = None,
):
    batch_size = (batch_size or config.learning_config.running_config.batch_size) * strategy.num_replicas_in_sync

    max_input_length, max_label_length = 0, 0
    for dset in datasets:
        max_input_length = max(max_input_length, dset.max_input_length or 0)
        max_label_length = max(max_label_length, dset.max_label_length or 0)
    max_input_length = None if max_input_length == 0 else max_input_length
    max_label_length = None if max_label_length == 0 else max_label_length

    input_shape = [max_input_length]
    prediction_shape = [max_label_length + 1] if max_label_length else [None]
    label_shape = [max_label_length]
    padded_shapes = schemas.TrainData(
        inputs=schemas.TrainInput(
            inputs=tf.TensorShape(input_shape),
            inputs_length=tf.TensorShape([]),
            predictions=tf.TensorShape(prediction_shape),
            predictions_length=tf.TensorShape([]),
        ),
        labels=schemas.TrainLabel(
            labels=tf.TensorShape(label_shape),
            labels_length=tf.TensorShape([]),
        ),
    )

    model_shapes = dict(
        batch_size=batch_size,
        input_shape=input_shape,
        prediction_shape=prediction_shape,
    )
    return model_shapes, batch_size, padded_shapes


BUFFER_SIZE = 100
TFRECORD_BUFFER_SIZE = 32 * 1024 * 1024
TFRECORD_SHARDS = 16
AUTOTUNE = int(os.environ.get("AUTOTUNE") or tf.data.AUTOTUNE)


# Where `scripts/install_kenlm.sh` leaves the binary, relative to the project root.
KENLM_LMPLZ = os.path.join("externals", "kenlm", "build", "bin", "lmplz")


def _resolve_lmplz(lmplz: str = None) -> str:
    """
    Find KenLM's `lmplz`, or say how to get it.

    An explicit path is taken as given -- a typo should fail rather than quietly fall back to some
    other copy and build a model nobody meant to build. Unset, PATH wins and the location
    `scripts/install_kenlm.sh` uses is the fallback, so a standard install needs no configuration.
    """
    if lmplz:
        found = shutil.which(lmplz)
        if found or os.path.isfile(lmplz):
            return found or lmplz
        raise FileNotFoundError(f"`{lmplz}` not found. Build it with ./scripts/install_kenlm.sh, or correct the path.")

    for candidate in (shutil.which("lmplz"), os.path.join(os.getcwd(), KENLM_LMPLZ)):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "`lmplz` not found on PATH or at ./"
        + KENLM_LMPLZ
        + ". Build it with ./scripts/install_kenlm.sh (run from the project root), or pass the path explicitly."
    )


class ASRDataset(AbstractDataset):
    def __init__(
        self,
        stage: str,
        tokenizer: AbstractTokenizer,
        data_paths: list,
        tfrecords_dir: str = None,
        tfrecords_shards: int = TFRECORD_SHARDS,
        tfrecords_buffer_size: int = TFRECORD_BUFFER_SIZE,
        tfrecords_compression_type: str = "GZIP",
        item_mapping: dict = None,
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
        self.tokenizer = tokenizer
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
        self.use_ga = False
        self.name = name or stage
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0:
            raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        self.tfrecords_buffer_size = tfrecords_buffer_size
        self.tfrecords_compression_type = tfrecords_compression_type
        self.item_mapping = item_mapping or {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.entries = []
        self.max_input_length = None
        self.max_label_length = None
        self.load_metadata()

    # -------------------------------- metadata -------------------------------------

    def compute_metadata(self):
        if not self.tokenizer.initialized:
            raise ValueError("Tokenizer must be initialized before computing metadata")

        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        self.max_input_length = 0 if self.max_input_length is None else self.max_input_length
        self.max_label_length = 0 if self.max_label_length is None else self.max_label_length
        self.read_entries()
        for _, duration, transcript in tqdm(self.entries, desc=f"Computing metadata for entries in {self.stage} dataset", disable=False):
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

    def vocab_generator(self):
        for *_, transcript in self.entries:
            yield transcript

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def generator(self):
        for path, _, transcript in self.entries:
            audio = data_util.load_and_convert_to_wav(path, sample_rate=self.sample_rate).numpy()
            yield bytes(path, "utf-8"), audio, bytes(transcript, "utf-8")

    def _process_item(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        with tf.device("/CPU:0"):
            inputs = data_util.read_raw_audio(audio)
            inputs_length = tf.shape(inputs, out_type=tf.int32)[0]

            labels = self.tokenizer.tokenize(transcript)
            labels_length = tf.shape(labels, out_type=tf.int32)[0]

            predictions = self.tokenizer.prepand_blank(labels)
            predictions_length = tf.shape(predictions, out_type=tf.int32)[0]

            return path, inputs, inputs_length, labels, labels_length, predictions, predictions_length

    def parse(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor) -> schemas.TrainData:
        (
            _,
            inputs,
            inputs_length,
            labels,
            labels_length,
            predictions,
            predictions_length,
        ) = self._process_item(path=path, audio=audio, transcript=transcript)
        return schemas.TrainData(
            inputs=schemas.TrainInput(inputs=inputs, inputs_length=inputs_length, predictions=predictions, predictions_length=predictions_length),
            labels=schemas.TrainLabel(labels=labels, labels_length=labels_length),
        )

    # -------------------------------- CREATION -------------------------------------

    def process(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        ga_steps: int = 1,
        padded_shapes=None,
    ):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE, deterministic=False)

        if self.cache:
            dataset = dataset.cache()  # cache original (unchanged data)

        if self.shuffle:
            dataset = dataset.shuffle(max(self.buffer_size or self.num_entries, batch_size * 2), reshuffle_each_iteration=True)

        if self.indefinite and hasattr(self, "total_steps") and self.total_steps:
            dataset = dataset.repeat()

        if padded_shapes is None:
            padded_shapes = schemas.TrainData(
                inputs=schemas.TrainInput(
                    inputs=tf.TensorShape([self.max_input_length]),
                    inputs_length=tf.TensorShape([]),
                    predictions=tf.TensorShape([self.max_label_length + 1 if self.max_label_length else None]),
                    predictions_length=tf.TensorShape([]),
                ),
                labels=schemas.TrainLabel(
                    labels=tf.TensorShape([self.max_label_length]),
                    labels_length=tf.TensorShape([]),
                ),
            )

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            padding_values=schemas.TrainData(
                inputs=schemas.TrainInput(inputs=0.0, inputs_length=0, predictions=self.tokenizer.blank, predictions_length=0),
                labels=schemas.TrainLabel(labels=self.tokenizer.blank, labels_length=0),
            ),
            drop_remainder=self.drop_remainder,
        )

        # only apply for training dataset, eval and test dataset should not use GA
        if ga_steps > 1 and self.stage == "train":
            self.use_ga = True

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)

        # Update metadata
        if hasattr(self, "num_entries") and self.num_entries > 0:
            self.total_steps = math_util.get_num_batches(self.num_entries, batch_size, drop_remainders=self.drop_remainder)
            if self.use_ga:
                self.total_steps = math_util.get_num_batches(self.total_steps, ga_steps, drop_remainders=False)

        return dataset

    def create(self, batch_size: int, ga_steps: int = 1, padded_shapes=None):
        if not self.enabled:
            return None
        if not self.tokenizer.initialized:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
        )
        return self.process(dataset, batch_size, ga_steps=ga_steps, padded_shapes=padded_shapes)


class ASRTFRecordDataset(ASRDataset):
    """Dataset for ASR using TFRecords"""

    def write_tfrecord_file(self, splitted_entries: tuple):
        shard_path, entries = splitted_entries
        logger.info(f"Processing {shard_path} ...")
        with tf.io.TFRecordWriter(shard_path, options=tf.io.TFRecordOptions(compression_type=self.tfrecords_compression_type)) as writer:
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

    def create(self, batch_size: int, ga_steps: int = 1, padded_shapes=None):
        if not self.enabled:
            return None
        if not self.tokenizer.initialized:
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
            files_ds,
            compression_type=self.tfrecords_compression_type,
            buffer_size=self.tfrecords_buffer_size,
            num_parallel_reads=AUTOTUNE,
        )

        return self.process(dataset, batch_size, ga_steps=ga_steps, padded_shapes=padded_shapes)


class ASRSliceDataset(ASRDataset):
    """Dataset for ASR using Slice"""

    def load(self, record):
        audio = tf.numpy_function(
            lambda path: data_util.load_and_convert_to_wav(path.decode("utf-8"), sample_rate=self.sample_rate).numpy(),
            inp=[record[0]],
            Tout=tf.string,
        )
        return record[0], audio, record[2]

    def create(self, batch_size: int, ga_steps: int = 1, padded_shapes=None):
        if not self.enabled:
            return None
        if not self.tokenizer.initialized:
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

        return self.process(dataset, batch_size, ga_steps=ga_steps, padded_shapes=padded_shapes)


class LMDataset(AbstractDataset):
    """
    Text dataset for training a `LanguageModel` (see `scripts/train_lm.py`).

    `data_paths` may mix two kinds of file, told apart by extension:

    - `.tsv` -- ASR transcript files (`PATH\tDURATION\tTRANSCRIPT`, with the header row skipped), of
      which only the transcript column is read. This is the text the transducer trained on, and the
      right source for the *internal* LM that LODR subtracts.
    - `.txt` / `.txt.gz` -- a plain text corpus, one sentence per line, read directly through gzip
      when the name ends in `.gz` (how OpenSLR ships the LibriSpeech LM corpus). This is the right
      source for the *external* LM, whose whole point is far more text than the transcripts -- the
      LibriSpeech LM corpus is ~40M lines and several GB.

    Everything streams one line at a time -- nothing is held in memory -- so a multi-GB external
    corpus is fine. Tokenisation uses the same tokenizer built from the config, which is what
    guarantees the indices match the transducer's vocabulary (the requirement
    `LanguageModel.call_next` states but cannot check).

    `max_input_length` is the longest tokenised sequence in the corpus. For a language model the
    text *is* the input (input and target are the same sequence shifted by one, see `shift_tokens`),
    so this is the LM analog of `ASRDataset.max_label_length`, and it is computed, saved and loaded
    the same way -- a JSON file keyed by `stage`.
    """

    def __init__(
        self,
        stage: str,
        tokenizer: AbstractTokenizer,
        data_paths: list,
        metadata: str = None,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        cache: bool = False,
        drop_remainder: bool = True,
        indefinite: bool = True,
        enabled: bool = True,
        max_length: int = 0,  # truncate sequences to this many tokens; 0 = no limit
        max_lines: int = None,  # stop after this many lines across all files; None = read all
        name: str = "",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.stage = stage
        self.metadata = metadata
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.cache = cache
        self.drop_remainder = drop_remainder
        self.indefinite = indefinite
        self.enabled = enabled
        self.max_length = max_length
        self.max_lines = max_lines
        self.name = name or stage
        self.total_steps = None
        self.num_entries = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.max_input_length = None
        self.load_metadata()

    # -------------------------------- FILES -------------------------------------

    def _resolved_paths(self):
        return file_util.preprocess_paths(self.data_paths, enabled=self.enabled, check_exists=True) or []

    def _iter_tsv(self, path):
        with tf.io.gfile.GFile(path, "r") as f:
            for line in f.read().splitlines()[1:]:  # skip the header of the tsv file
                parts = line.split("\t", 2)  # PATH \t DURATION \t TRANSCRIPT
                if len(parts) == 3:
                    yield parts[2]

    def _iter_textfile(self, path):
        if path.lower().endswith(".gz"):
            # gfile does not decompress, so wrap its bytes stream in gzip -- this keeps remote
            # paths (gs://) working, which a plain gzip.open would not.
            with tf.io.gfile.GFile(path, "rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
                for line in gz:
                    yield line.decode("utf-8").rstrip("\r\n")
        else:
            with tf.io.gfile.GFile(path, "r") as f:
                for line in f:
                    yield line.rstrip("\r\n")

    def _iter_texts(self):
        """Stream text, one line at a time, across every file in `data_paths`."""
        count = 0
        for path in self._resolved_paths():
            source = self._iter_tsv(path) if path.lower().endswith(".tsv") else self._iter_textfile(path)
            for text in source:
                yield text
                count += 1
                if self.max_lines and count >= self.max_lines:
                    return

    # -------------------------------- ENTRIES -------------------------------------

    def read_entries(self):
        # Only resolves and validates the paths; unlike the ASR dataset it does not read them, so a
        # multi-GB corpus is not walked here. `num_entries` stays 0 (unknown) until metadata is
        # computed -- the size of the external corpus is why `train_lm.py` asks for
        # `--steps-per-epoch` rather than counting.
        self.data_paths = self._resolved_paths()
        for path in self.data_paths:
            logger.info(f"Using LM text from {path} ...")

    def generator(self):
        for text in self._iter_texts():
            yield bytes(text, "utf-8")

    def vocab_generator(self):
        for text in self._iter_texts():
            yield text

    def token_generator(self):
        """
        Tokenized text, one numpy array per line, for the n-gram `fit_counts` counting pass.

        `fit_counts` tallies bigrams in python and has no use for a `tf.data` stream, so this yields
        eagerly. The gradient-descent path uses `create` instead.
        """
        for text in self._iter_texts():
            yield self.tokenizer.tokenize(text).numpy()

    # -------------------------------- KENLM -------------------------------------

    def write_token_ids(self, path, max_lines: int = None):
        """
        Write this corpus as space-separated token ids, one sentence per line.

        The form `lmplz` needs. It has no idea what a word-piece is, so tokens go out as **integer
        ids** -- the ids of the very tokenizer this dataset holds. That is also what guarantees the
        resulting language model is indexed against the same vocabulary the transducer emits, which
        is the one thing `NGramLanguageModel.call_next` requires and cannot check.

        A `.gz` suffix is written gzipped. Returns `(lines, tokens)`.
        """
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        path = file_util.preprocess_paths(path)
        opener = gzip.open if str(path).endswith(".gz") else open
        lines = tokens = 0
        with opener(path, "wt", encoding="utf-8") as handle:
            logger.info(f"Writing token ids to {path} ...")
            for ids in tqdm(self.token_generator(), desc="Writing", unit=" lines", disable=False):
                ids = [int(i) for i in ids]
                if not ids:
                    continue  # lmplz reads an empty line as a sentence, which would skew the counts
                handle.write(" ".join(map(str, ids)))
                handle.write("\n")
                lines += 1
                tokens += len(ids)
                if max_lines and lines >= max_lines:
                    break
        return lines, tokens

    def create_arpa(
        self,
        arpa_path,
        text_path=None,
        order: int = 4,
        prune=None,
        max_lines: int = None,
        lmplz: str = None,
        lmplz_args=None,
        overwrite_text: bool = False,
    ):
        """
        Build a token-level ARPA n-gram over this corpus with KenLM, and return its path.

        Two steps: tokenise the corpus to token ids (`write_token_ids`), then hand that to `lmplz`.
        The intermediate text is kept rather than piped, because tokenising a large corpus is the
        slow half and keeping it lets `lmplz` be re-run at another order or pruning without paying
        for it twice.

        `lmplz` must be on PATH -- `scripts/install_kenlm.sh` builds it and says how. This is the
        only path to an n-gram over a corpus of any size: `NGramLanguageModel.fit_counts` counts in
        python dicts and tops out around a few million tokens.

        Parameters
        ----------
        arpa_path : str
            Where to write the ARPA.
        text_path : Optional[str]
            Where to write the token-id text. Defaults to `<arpa_path without suffix>.ids.txt`. An
            existing file is **reused**, not rebuilt, so a second run at a different order skips the
            tokenisation. See `overwrite_text` for when that is the wrong thing.
        order : int
            `lmplz -o`. Must match `NGramLanguageModel`'s `order`.
        prune : Optional[Sequence[int]]
            `lmplz --prune`, one count cutoff per order. This is where to shrink the model; it is
            cheaper and better informed than having `read_arpa` drop arcs to fit its budget.
        max_lines : Optional[int]
            Stop after this many lines, for a trial run over a huge corpus.
        lmplz : Optional[str]
            The binary. Left unset it is looked for on PATH and then at `KENLM_LMPLZ`, where
            `scripts/install_kenlm.sh` puts it, so a standard install needs nothing here. An explicit
            value is used as given and never falls back, so a typo fails loudly instead of silently
            building with some other copy.
        lmplz_args : Optional[Sequence[str]]
            Extra flags passed through, eg. `["-S", "40%"]` to cap memory. `--discount_fallback` is
            **not** needed here -- it is added automatically and the build retried if `lmplz` aborts
            because an order has too few n-grams to estimate a discount, which is routine for a
            token-level corpus. Pass it explicitly only to force fallback discounts on the first try.
            `lmplz`'s scratch (`-T`) is placed next to the ARPA output and cleaned up, rather than
            left on its `/tmp` default where a real corpus's tens of GB of merge-sort spill can fill
            the disk; pass your own `-T`/`--temp_prefix` here to override that.
        overwrite_text : bool
            Re-tokenise even when `text_path` already exists. Reuse is the default because
            tokenising is the slow half and it is usually what you want between runs that only
            change `order` or `prune`. It is **wrong** whenever the ids would come out different:
            the corpus changed, `data_paths` changed, `max_lines` changed, or the tokenizer did.
            Nothing detects that -- a stale file is still a valid file -- so an LM built over the
            previous corpus, or worse over a previous *vocabulary*, would be indexed against
            indices the transducer no longer emits. Pass this when in doubt; the cost is time.
        """
        lmplz = _resolve_lmplz(lmplz)

        arpa_path = file_util.preprocess_paths(arpa_path)
        if text_path is None:
            text_path = os.path.splitext(str(arpa_path))[0] + ".ids.txt"
        text_path = file_util.preprocess_paths(text_path)

        reusable = os.path.isfile(text_path) and os.path.getsize(text_path) > 0
        if reusable and not overwrite_text:
            logger.info(f"Reusing the token ids already at {text_path}; pass overwrite_text=True (--overwrite-text) to rebuild")
        else:
            if reusable:
                logger.info(f"Overwriting the token ids at {text_path}")
            lines, tokens = self.write_token_ids(text_path, max_lines=max_lines)
            if not lines:
                raise ValueError(f"No usable text in {self.data_paths}; nothing to build a language model from.")
            logger.info(f"Wrote {lines:,} lines / {tokens:,} tokens to {text_path}")

        command = [lmplz, "-o", str(order)]
        if prune:
            command += ["--prune", *[str(int(p)) for p in prune]]
        if lmplz_args:
            command += [str(a) for a in lmplz_args]

        # Keep `lmplz`'s scratch off `/tmp`. Its merge sort spills the sorted n-gram counts to the
        # `-T` prefix, which defaults to `/tmp` -- tens of GB for a real corpus, and `/tmp` is the
        # small root filesystem on a Kaggle box (the model dir is the roomy one). Put the scratch on
        # the same volume as the ARPA output instead, and delete it afterwards so it is never left
        # behind or, on Kaggle, swept up as notebook output. Skipped if the caller set `-T` itself.
        temp_dir = None
        if not any(flag in command for flag in ("-T", "--temp_prefix")):
            temp_dir = os.path.join(os.path.dirname(arpa_path) or ".", ".lmplz_tmp")
            os.makedirs(temp_dir, exist_ok=True)
            command += ["-T", os.path.join(temp_dir, "")]  # trailing sep: files land inside the dir

        def run(cmd):
            """Run `lmplz`, corpus in on stdin, ARPA out to the file. stderr stays live so a long build is visible."""
            logger.info(f"Running {' '.join(cmd)} < {text_path} > {arpa_path}")
            with open(text_path, "rb") as source, open(arpa_path, "wb") as destination:
                return subprocess.run(cmd, stdin=source, stdout=destination, check=False).returncode

        try:
            returncode = run(command)
            if returncode != 0 and "--discount_fallback" not in command:
                # `lmplz` aborts (SIGABRT, exit -6) when an n-gram order has too few distinct counts
                # to estimate a Kneser-Ney discount -- routine for a token-level model, where the
                # vocabulary is small and the higher orders are sparse, which is the only kind this
                # ever builds. `--discount_fallback` substitutes default discounts for the orders
                # that cannot be estimated and leaves the rest alone, so retrying with it is the
                # standard fix and costs only a second pass over a file already on disk. Added on
                # failure rather than always, so a corpus healthy enough to estimate real discounts
                # still gets them.
                logger.warning(
                    f"lmplz exited {returncode}; retrying with --discount_fallback. An order had too few n-grams to estimate "
                    f"a discount, normal for a token-level corpus -- the model will use fallback discounts for those orders."
                )
                command.append("--discount_fallback")
                returncode = run(command)
            if returncode != 0:
                raise RuntimeError(
                    f"lmplz exited {returncode}. If this was already retried with --discount_fallback, the corpus is likely "
                    f"too small or too repetitive to build an n-gram of this order; try a lower order or more text. "
                    f"Its own output above says which stage failed."
                )
        finally:
            # Even on failure `lmplz` can leave partial spill files behind; a second run would then
            # build over gigabytes of stale scratch.
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Wrote {arpa_path}")
        return arpa_path

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def _text_line_dataset(self):
        """A streaming `tf.data.Dataset` of text lines (string tensors) over all files."""
        line_datasets = []
        for path in self._resolved_paths():
            lower = path.lower()
            if lower.endswith(".tsv"):
                # Transcripts are bounded (the ASR training set), so materialising them is cheap and
                # gives a replayable dataset -- a generator walked once would leave later passes empty.
                transcripts = list(self._iter_tsv(path))
                if transcripts:
                    line_datasets.append(tf.data.Dataset.from_tensor_slices(transcripts))
            else:
                # TextLineDataset streams the file in C++ (transparently through gzip for `.gz`), so
                # a multi-GB corpus is never read into memory; tokenization runs in the graph after.
                line_datasets.append(
                    tf.data.TextLineDataset(
                        path,
                        compression_type="GZIP" if lower.endswith(".gz") else "",
                        num_parallel_reads=AUTOTUNE,
                    )
                )
        if not line_datasets:
            raise ValueError(f"No readable data files in data_paths for the {self.stage} LM dataset")
        dataset = line_datasets[0]
        for extra in line_datasets[1:]:
            dataset = dataset.concatenate(extra)
        return dataset

    def _token_dataset(self):
        """A streaming `tf.data.Dataset` of tokenized `int32` vectors, one per line."""
        dataset = self._text_line_dataset()
        if self.max_lines:
            dataset = dataset.take(self.max_lines)
        return dataset.map(lambda line: tf.cast(self.tokenizer.tokenize(line), tf.int32), num_parallel_calls=AUTOTUNE)

    def create(self, batch_size: int):
        """
        A batched `(inputs, targets, sample_weight)` dataset for training a `LanguageModel`.

        Targets are the token sequence; inputs are the same sequence shifted right with blank in
        front, so position `u` predicts token `u` from everything before it -- the conditioning the
        beam search hands to `call_next`. `sample_weight` is 1 on real tokens and 0 on padding, built
        *before* batching as an all-ones vector and then padded with 0: blank is the pad value *and*
        a legal token (it doubles as start of sentence), so the mask cannot be recovered afterwards.

        Every knob is read from the config on `self`:

        - `max_length` truncates each sequence, and is also the padded length when set -- a fixed
          shape, which is what XLA/TPU needs. Left unset (0), batches pad to their own longest
          sequence, which is cheaper on a GPU. Pinning the shape for a TPU is therefore just
          `max_length` plus `drop_remainder` in the LM dataset config, the same way the ASR datasets
          pin from metadata.
        - `shuffle` with `buffer_size` shuffles single sequences before batching, so the model does
          not spend thousands of consecutive steps inside one slice of a corpus laid out by document.
        - `drop_remainder` drops the short final batch (needed on TPU, where every batch must share
          one shape).
        - `indefinite` repeats the data so a finite corpus can fill fixed-size epochs. It is applied
          *after* batching, so one cycle is exactly `ceil(sequences / batch_size)` batches rather
          than letting a batch straddle the seam.
        """
        from tensorflow_asr.models.lm.language_model import shift_tokens  # pylint: disable=import-outside-toplevel

        blank = self.tokenizer.blank
        max_length = self.max_length or None  # 0 => no truncation; t[:None] keeps the whole sequence
        sequence_shape = [self.max_length] if self.max_length else [None]

        dataset = self._token_dataset()
        dataset = dataset.map(lambda t: t[:max_length], num_parallel_calls=AUTOTUNE)
        dataset = dataset.filter(lambda t: tf.size(t) > 0)  # blank lines carry no supervision
        if self.shuffle and self.buffer_size:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.map(lambda t: (t, tf.ones_like(t, dtype=tf.float32)), num_parallel_calls=AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(sequence_shape, sequence_shape),
            padding_values=(tf.constant(blank, tf.int32), 0.0),
            drop_remainder=self.drop_remainder,
        )
        dataset = dataset.map(lambda t, w: (shift_tokens(t, blank), t, w), num_parallel_calls=AUTOTUNE)
        if self.indefinite:
            dataset = dataset.repeat()
        return dataset.prefetch(AUTOTUNE)

    # -------------------------------- metadata -------------------------------------

    def compute_metadata(self):
        if not self.tokenizer.initialized:
            raise ValueError("Tokenizer must be initialized before computing metadata")

        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        self.max_input_length = 0 if self.max_input_length is None else self.max_input_length
        num_entries = 0
        for tokens in tqdm(self.token_generator(), desc=f"Computing metadata for entries in {self.stage} LM dataset", disable=False):
            self.max_input_length = max(self.max_input_length, len(tokens))
            num_entries += 1
        self.total_steps = num_entries
        self.num_entries = num_entries

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
        self.total_steps = int(content.get("num_entries", 0))
        self.num_entries = self.total_steps

    def update_metadata(self):
        self.load_metadata()
        self.compute_metadata()
        self.save_metadata()


def get_lm(
    tokenizer: AbstractTokenizer,
    dataset_config: DatasetConfig,
    **overrides,
):
    """
    Build an `LMDataset` from a `DatasetConfig`, mirroring `get` for the ASR datasets.

    `overrides` win over the config, which is how `scripts/train_lm.py` points the external LM at
    `--text-path` (`data_paths=[text_path]`) and passes `max_length` / `max_lines`.
    """
    return LMDataset(tokenizer=tokenizer, **{**vars(dataset_config), **overrides})
