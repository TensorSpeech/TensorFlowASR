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

import json
from typing import Union

import tensorflow as tf

from tensorflow_asr.utils import file_util

logger = tf.get_logger()


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
        self.character_coverage: float = config.pop("character_coverage", 1.0)  # 0.9995 for languages with rich character, else 1.0

        self.train_files = config.pop("train_files", [])
        self.eval_files = config.pop("eval_files", [])

        for k, v in config.items():
            setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.name: str = config.pop("name", "")
        self.enabled: bool = config.pop("enabled", True)
        self.stage: str = config.pop("stage", None)
        self.data_paths = config.pop("data_paths", None)
        self.tfrecords_dir: str = config.pop("tfrecords_dir", None)
        self.tfrecords_shards: int = config.pop("tfrecords_shards", 16)
        self.tfrecords_buffer_size: int = config.pop("tfrecords_buffer_size", 32 * 1024 * 1024)
        self.shuffle: bool = config.pop("shuffle", False)
        self.cache: bool = config.pop("cache", False)
        self.drop_remainder: bool = config.pop("drop_remainder", True)
        self.buffer_size: int = config.pop("buffer_size", 1000)
        self.metadata: str = config.pop("metadata", None)
        self.sample_rate: int = config.pop("sample_rate", 16000)
        for k, v in config.items():
            setattr(self, k, v)


class DataConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_configs = [DatasetConfig(conf) for conf in config.pop("test_dataset_configs", [])]
        _test_dataset_config = config.pop("test_dataset_config", None)
        if _test_dataset_config:
            self.test_dataset_configs.append(_test_dataset_config)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.pretrained = file_util.preprocess_paths(config.pop("pretrained", None))
        self.optimizer_config: dict = config.pop("optimizer_config", {})
        self.gwn_config = config.pop("gwn_config", None)
        self.gradn_config = config.pop("gradn_config", None)
        self.batch_size: int = config.pop("batch_size", 2)
        self.ga_steps: int = config.pop("ga_steps", None)
        self.num_epochs: int = config.pop("num_epochs", 300)
        self.callbacks: list = config.pop("callbacks", [])
        for k, v in config.items():
            setattr(self, k, v)


class Config:
    """User config class for training, testing or infering"""

    def __init__(self, data: Union[str, dict], training=True, **kwargs):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data), **kwargs)
        self.decoder_config = DecoderConfig(config.pop("decoder_config", {}))
        self.model_config: dict = config.pop("model_config", {})
        self.data_config = DataConfig(config.pop("data_config", {}))
        self.learning_config = LearningConfig(config.pop("learning_config", {})) if training else None
        for k, v in config.items():
            setattr(self, k, v)
        logger.info(str(self))

    def __str__(self) -> str:
        def default(x):
            try:
                return {k: v for k, v in vars(x).items() if not str(k).startswith("_")}
            except:  # pylint: disable=bare-except
                return str(x)

        return json.dumps(vars(self), indent=2, default=default)
