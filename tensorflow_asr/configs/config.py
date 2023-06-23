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

from typing import Union

import tensorflow as tf

from tensorflow_asr.utils import file_util


class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.type: str = config.pop("type", "wordpiece")

        self.blank_index: int = config.pop("blank_index", 0)
        self.pad_token: str = config.pop("pad_token", "<pad>")
        self.pad_index: int = config.pop("pad_index", -1)
        self.unknown_token: str = config.pop("unknown_token", "<unk>")
        self.unknown_index: int = config.pop("unknown_index", 0)
        self.bos_token: str = config.pop("bos_token", "<s>")
        self.bos_index: int = config.pop("bos_index", -1)
        self.eos_token: str = config.pop("eos_token", "</s>")
        self.eos_index: int = config.pop("eos_index", -1)

        self.beam_width: int = config.pop("beam_width", 0)
        self.norm_score: bool = config.pop("norm_score", True)
        self.lm_config: dict = config.pop("lm_config", {})

        self.model_type: str = config.pop("model_type", "unigram")
        self.vocabulary: str = file_util.preprocess_paths(config.pop("vocabulary", None))
        self.vocab_size: int = config.pop("vocab_size", 1000)
        self.max_token_length: int = config.pop("max_token_length", 50)
        self.max_unique_chars: int = config.pop("max_unique_chars", None)
        self.num_iterations: int = config.pop("num_iterations", 4)
        self.reserved_tokens: list = config.pop("reserved_tokens", None)
        self.normalization_form: str = config.pop("normalization_form", "NFKC")
        self.keep_whitespace: bool = config.pop("keep_whitespace", False)
        self.max_sentence_length: int = config.pop("max_sentence_length", 1048576)  # bytes
        self.max_sentencepiece_length: int = config.pop("max_sentencepiece_length", 16)  # bytes

        self.train_files = file_util.preprocess_paths(config.pop("train_files", []))
        self.eval_files = file_util.preprocess_paths(config.pop("eval_files", []))

        for k, v in config.items():
            setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.enabled: bool = config.pop("enabled", True)
        self.stage: str = config.pop("stage", None)
        self.data_paths = file_util.preprocess_paths(config.pop("data_paths", None), enabled=self.enabled)
        self.tfrecords_dir: str = file_util.preprocess_paths(config.pop("tfrecords_dir", None), isdir=True, enabled=self.enabled)
        self.tfrecords_shards: int = config.pop("tfrecords_shards", 16)
        self.shuffle: bool = config.pop("shuffle", False)
        self.cache: bool = config.pop("cache", False)
        self.drop_remainder: bool = config.pop("drop_remainder", True)
        self.buffer_size: int = config.pop("buffer_size", 1000)
        self.metadata: str = config.pop("metadata", None)
        self.sample_rate: int = config.pop("sample_rate", 16000)
        for k, v in config.items():
            setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.batch_size: int = config.pop("batch_size", 2)
        self.ga_steps: int = config.pop("ga_steps", None)
        self.num_epochs: int = config.pop("num_epochs", 100)
        self.checkpoint: dict = {}
        self.backup_and_restore: dict = {}
        self.tensorboard: dict = {}
        self.early_stopping: dict = {}
        for k, v in config.items():
            setattr(self, k, v)
            if k == "checkpoint":
                if v and v.get("filepath"):
                    file_util.preprocess_paths(v.get("filepath"))
                if v and v.get("options"):
                    self.checkpoint["options"] = tf.train.CheckpointOptions(**v.get("options"))
            elif k == "backup_and_restore" and v:
                if v and v.get("backup_dir"):
                    file_util.preprocess_paths(v.get("backup_dir"), isdir=True)
            elif k == "tensorboard":
                if v and v.get("log_dir"):
                    file_util.preprocess_paths(v.get("log_dir"), isdir=True)


class DataConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.pretrained = file_util.preprocess_paths(config.pop("pretrained", None))
        self.optimizer_config: dict = config.pop("optimizer_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        self.apply_gwn_config = config.pop("apply_gwn_config", None)
        for k, v in config.items():
            setattr(self, k, v)


class Config:
    """User config class for training, testing or infering"""

    def __init__(self, data: Union[str, dict]):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data))
        self.decoder_config = DecoderConfig(config.pop("decoder_config", {}))
        self.model_config: dict = config.pop("model_config", {})
        self.data_config = DataConfig(config.pop("data_config", {}))
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items():
            setattr(self, k, v)
