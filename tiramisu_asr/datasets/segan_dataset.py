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
import glob
import os

import tensorflow as tf

from ..augmentations.augments import Noise
from ..datasets.base_dataset import BaseDataset
from ..featurizers.speech_featurizers import preemphasis, read_raw_audio
from ..utils.utils import slice_signal, append_default_keys_dict, merge_slices

DEFAULT_NOISE_CONF = {
    "snr_list":   (0, 5, 10, 15),
    "max_noises": 3,
    "include_original": True
}


class SeganDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 data_paths: list,
                 noises_config: dict,
                 speech_config: dict,
                 shuffle: bool = False):
        self.noises = Noise(append_default_keys_dict(DEFAULT_NOISE_CONF, noises_config))
        self.speech_config = speech_config
        super(SeganDataset, self).__init__(data_paths, None, shuffle, stage)

    def _merge_dirs(self):
        dirs = []
        for paths in self.data_paths:
            dirs += glob.glob(os.path.join(paths, "**", "*.wav"), recursive=True)
        return dirs

    def parse(self, record, stride=1):
        clean_wav = record
        noisy_wav = self.noises(clean_wav, sample_rate=self.speech_config["sample_rate"])

        clean_wav = preemphasis(clean_wav, self.speech_config["preemphasis"])
        noisy_wav = preemphasis(noisy_wav, self.speech_config["preemphasis"])

        clean_slices = slice_signal(clean_wav, self.speech_config["window_size"], stride)
        noisy_slices = slice_signal(noisy_wav, self.speech_config["window_size"], stride)

        return clean_slices, noisy_slices

    def create(self, batch_size):
        def _gen_data():
            for clean_wav_path in self._merge_dirs():
                clean_wav = read_raw_audio(clean_wav_path, sample_rate=self.speech_config["sample_rate"])
                clean_slices, noisy_slices = self.parse(clean_wav, self.speech_config["stride"])
                for clean, noisy in zip(clean_slices, noisy_slices):
                    yield clean, noisy

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(
                tf.float32,
                tf.float32
            ),
            output_shapes=(
                tf.TensorShape([self.speech_config["window_size"]]),
                tf.TensorShape([self.speech_config["window_size"]])
            )
        )
        if self.shuffle:
            dataset = dataset.shuffle(16, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def create_test(self):
        def _gen_data():
            for clean_wav_path in self._merge_dirs():
                clean_wav = read_raw_audio(clean_wav_path, sample_rate=self.speech_config["sample_rate"])
                clean_slices, noisy_slices = self.parse(clean_wav, 1)
                yield os.path.basename(clean_wav_path), merge_slices(clean_slices), noisy_slices

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(tf.string, tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None, self.speech_config["window_size"]]))
        )
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
