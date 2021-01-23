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

from . import load_yaml
from ..augmentations.augments import Augmentation
from ..utils.utils import preprocess_paths


class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.vocabulary = preprocess_paths(config.pop("vocabulary", None))
        self.beam_width = config.pop("beam_width", 0)
        self.blank_at_zero = config.pop("blank_at_zero", True)
        self.target_vocab_size = config.pop("target_vocab_size", 1024)
        self.max_subword_length = config.pop("max_subword_length", 4)
        self.norm_score = config.pop("norm_score", True)
        self.lm_config = config.pop("lm_config", {})
        self.output_path_prefix = preprocess_paths(config.pop("output_path_prefix", None))
        self.model_type = config.pop("model_type", None)
        self.corpus_files = config.pop("corpus_files", None)
        for k, v in config.items(): setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.train_paths = preprocess_paths(config.pop("train_paths", None))
        self.eval_paths = preprocess_paths(config.pop("eval_paths", None))
        self.test_paths = preprocess_paths(config.pop("test_paths", None))
        self.tfrecords_dir = preprocess_paths(config.pop("tfrecords_dir", None))
        for k, v in config.items(): setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.batch_size = config.pop("batch_size", 1)
        self.accumulation_steps = config.pop("accumulation_steps", 1)
        self.num_epochs = config.pop("num_epochs", 20)
        self.outdir = preprocess_paths(config.pop("outdir", None))
        self.log_interval_steps = config.pop("log_interval_steps", 500)
        self.save_interval_steps = config.pop("save_interval_steps", 500)
        self.eval_interval_steps = config.pop("eval_interval_steps", 1000)
        for k, v in config.items(): setattr(self, k, v)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.augmentations = Augmentation(config.pop("augmentations", {}))
        self.dataset_config = DatasetConfig(config.pop("dataset_config", {}))
        self.optimizer_config = config.pop("optimizer_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        for k, v in config.items(): setattr(self, k, v)


class Config:
    """ User config class for training, testing or infering """

    def __init__(self, path: str):
        config = load_yaml(preprocess_paths(path))
        self.speech_config = config.pop("speech_config", {})
        self.decoder_config = config.pop("decoder_config", {})
        self.model_config = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items(): setattr(self, k, v)
