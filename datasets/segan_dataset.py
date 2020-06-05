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
from __future__ import absolute_import

import glob
import os
import tensorflow as tf
from utils.utils import slice_signal, append_default_keys_dict
from featurizers.speech_featurizers import preemphasis, read_raw_audio
from augmentations.noise_augment import add_noise

DEFAULT_NOISE_CONF = {
    "snr": (-1, 0, 5, 10, 15),
    "max_noises": 3,
}


class SeganDataset:
    def __init__(self, clean_data_dir, noises_dir, noise_conf=DEFAULT_NOISE_CONF, window_size=2 ** 14, stride=0.5):
        assert os.path.exists(clean_data_dir) and os.path.exists(noises_dir)
        self.clean_data_dir = clean_data_dir
        self.noises_dir = glob.glob(os.path.join(noises_dir, "**", "*.wav"), recursive=True)
        self.window_size = window_size
        self.stride = stride
        self.noise_conf = append_default_keys_dict(DEFAULT_NOISE_CONF, noise_conf)

    def create(self, batch_size, coeff=0.97, sample_rate=16000, shuffle=True):
        assert os.path.isdir(self.clean_data_dir)

        def _gen_data():
            for clean_wav_path in glob.iglob(os.path.join(self.clean_data_dir, "**", "*.wav"), recursive=True):
                # clean_split = clean_wav_path.split('/')
                # noisy_split = self.noisy_data_dir.split('/')
                # clean_split = clean_split[len(noisy_split):]
                # noisy_split = noisy_split + clean_split
                # noisy_wav_path = '/' + os.path.join(*noisy_split)

                clean_wav = read_raw_audio(clean_wav_path, sample_rate=sample_rate)
                clean_slices = slice_signal(clean_wav, self.window_size, self.stride)

                # noisy_wav = read_raw_audio(noisy_wav_path, sample_rate=16000)
                noisy_wav = add_noise(clean_wav, self.noises_dir, snr_list=self.noise_conf["snr"],
                                      max_noises=self.noise_conf["max_noises"], sample_rate=sample_rate)
                noisy_slices = slice_signal(noisy_wav, self.window_size, self.stride)

                for clean_slice, noisy_slice in zip(clean_slices, noisy_slices):
                    if len(clean_slice) == 0:
                        continue
                    yield preemphasis(clean_slice, coeff), preemphasis(noisy_slice, coeff)

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(
                tf.float32,
                tf.float32
            ),
            output_shapes=(
                tf.TensorShape([self.window_size]),
                tf.TensorShape([self.window_size])
            )
        )
        if shuffle:
            dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def create_test(self, sample_rate=16000):
        if os.path.isdir(self.clean_data_dir):
            def _gen_data():
                for clean_wav_path in glob.iglob(os.path.join(self.clean_data_dir, "**", "*.wav"), recursive=True):
                    clean_wav = read_raw_audio(clean_wav_path, sample_rate=sample_rate)
                    noisy_wav = add_noise(clean_wav, self.noises_dir, snr_list=self.noise_conf["snr"],
                                          max_noises=self.noise_conf["max_noises"], sample_rate=sample_rate)
                    yield clean_wav, noisy_wav
        else:
            with open(self.clean_data_dir, "r", encoding="utf-8") as en:
                entries = en.read().splitlines()
                entries = entries[1:]

            def _gen_data():
                for clean_wav_path in entries:
                    clean_wav_path = clean_wav_path.split("\t")[0]
                    clean_wav = read_raw_audio(clean_wav_path, sample_rate=sample_rate)
                    noisy_wav = add_noise(clean_wav, self.noises_dir, snr_list=self.noise_conf["snr"],
                                          max_noises=self.noise_conf["max_noises"], sample_rate=sample_rate)
                    yield clean_wav, noisy_wav

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(
                tf.float32,
                tf.float32
            ),
            output_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None])
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
