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
import os
import sys
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

import configs.user_config as config
from datasets.asr_dataset import ASRTFRecordDataset
from configs.user_config import UserConfig
from featurizers.speech_featurizers import SpeechFeaturizer
from featurizers.text_featurizers import TextFeaturizer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(config.__file__)), "default_ctc.yml")

config = UserConfig(DEFAULT_YAML, sys.argv[1], learning=True)
speech_featurizer = SpeechFeaturizer(config["speech_config"])
text_featurizer = TextFeaturizer(config["decoder_config"]["vocabulary"])

process = psutil.Process(os.getpid())

train_dataset = ASRTFRecordDataset(
    config["learning_config"]["dataset_config"]["train_paths"],
    config["learning_config"]["dataset_config"]["tfrecords_dir"],
    speech_featurizer, text_featurizer, "train",
    augmentations=config["learning_config"]["augmentations"], shuffle=True,
).create(config["learning_config"]["batch_size"])

for idx, batch in train_dataset.enumerate(start=1):
    print(f"Iter: {idx}, Memory: {process.memory_info().rss / (2 ** 20)}", end="\r")
