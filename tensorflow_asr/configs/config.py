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

from typing import Union

import tensorflow as tf

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.utils import file_util


class SpeechConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}

        # Sample rate in Hz
        self.sample_rate: int = config.pop("sample_rate", 16000)
        # Amount of data grabbed for each frame during analysis
        self.frame_ms: int = config.pop("frame_ms", 25)
        self.frame_length = int(round(self.sample_rate * self.frame_ms / 1000.0))
        # Number of ms to jump between frames
        self.stride_ms: int = config.pop("stride_ms", 10)
        self.frame_step = int(round(self.sample_rate * self.stride_ms / 1000.0))
        # Number of bins in the feature output
        self.num_feature_bins: int = config.pop("num_feature_bins", 80)
        # Type of feature extraction
        self.feature_type: str = config.pop("feature_type", "log_mel_spectrogram")

        # The first-order filter coefficient used for preemphasis. When it is 0.0, preemphasis is turned off.
        self.preemphasis: float = config.pop("preemphasis", 0.97)
        # Whether to pad the end of `signals` with zeros when framing produces a frame that lies partially past its end.
        self.pad_end: bool = config.pop("pad_end", False)
        # Use librosa like stft
        self.use_librosa_like_stft: bool = config.pop("use_librosa_like_stft", False)
        # Whether to use twice the minimum fft resolution.
        self.fft_overdrive: bool = config.pop("fft_overdrive", True)
        # Whether to compute filterbank output on the energy of spectrum rather than just the magnitude.
        self.compute_energy: bool = config.pop("compute_energy", False)
        # Minimum output of filterbank output prior to taking logarithm.
        self.output_floor: float = config.pop("output_floor", 1e-10)
        # Use natural log
        self.use_natural_log: bool = config.pop("use_natural_log", True)
        # The lowest frequency of the feature analysis
        self.lower_edge_hertz: float = config.pop("lower_edge_hertz", 125.0)
        # The highest frequency of the feature analysis
        self.upper_edge_hertz: float = config.pop("upper_edge_hertz", self.sample_rate / 2)

        self.normalize_signal: bool = config.pop("normalize_signal", False)
        self.normalize_feature: bool = config.pop("normalize_feature", True)
        self.normalize_per_frame: bool = config.pop("normalize_per_frame", False)

        for k, v in config.items():
            setattr(self, k, v)


class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.type: str = config.pop("type", "wordpiece")

        self.blank_index: int = config.pop("blank_index", 0)
        self.pad_token: str = config.pop("pad_token", "<pad>")
        self.pad_index: int = config.pop("pad_index", 0)
        self.unknown_token: str = config.pop("unknown_token", "<unk>")
        self.unknown_index: int = config.pop("unknown_index", 1)
        self.bos_token: str = config.pop("bos_token", "<s>")
        self.bos_index: int = config.pop("bos_index", 2)
        self.eos_token: str = config.pop("eos_token", "</s>")
        self.eos_index: int = config.pop("eos_index", 3)

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

        self.corpus_files = file_util.preprocess_paths(config.pop("corpus_files", []))

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
        self.use_tf: bool = config.pop("use_tf", False)
        self.augmentations = Augmentation(config.pop("augmentation_config", {}))
        self.metadata: str = config.pop("metadata", None)
        for k, v in config.items():
            setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.batch_size: int = config.pop("batch_size", 1)
        self.accumulation_steps: int = config.pop("accumulation_steps", 1)
        self.num_epochs: int = config.pop("num_epochs", 20)
        self.checkpoint: dict = {}
        self.backup_and_restore: dict = {}
        self.tensorboard: dict = {}
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


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.pretrained = file_util.preprocess_paths(config.pop("pretrained", None))
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))
        self.optimizer_config: dict = config.pop("optimizer_config", {})
        self.learning_rate_config: dict = config.pop("learning_rate_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        self.apply_gwn_config = config.pop("apply_gwn_config", None)
        for k, v in config.items():
            setattr(self, k, v)


class Config:
    """User config class for training, testing or infering"""

    def __init__(self, data: Union[str, dict]):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data))
        self.speech_config = SpeechConfig(config.pop("speech_config", {}))
        self.decoder_config = DecoderConfig(config.pop("decoder_config", {}))
        self.model_config: dict = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items():
            setattr(self, k, v)
