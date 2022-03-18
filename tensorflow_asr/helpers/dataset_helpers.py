# Copyright 2022 Huy Le Nguyen (@usimarit)
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

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets import asr_dataset
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer


def prepare_training_datasets(
    config: Config,
    speech_featurizer: SpeechFeaturizer,
    text_featurizer: TextFeaturizer,
    tfrecords: bool = False,
):
    if tfrecords:
        train_dataset = asr_dataset.ASRTFRecordDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.train_dataset_config),
            indefinite=True
        )
        eval_dataset = asr_dataset.ASRTFRecordDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.eval_dataset_config),
            indefinite=True
        )
    else:
        train_dataset = asr_dataset.ASRSliceDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.train_dataset_config),
            indefinite=True
        )
        eval_dataset = asr_dataset.ASRSliceDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.eval_dataset_config),
            indefinite=True
        )
    return train_dataset, eval_dataset


def prepare_testing_datasets(
    config: Config,
    speech_featurizer: SpeechFeaturizer,
    text_featurizer: TextFeaturizer,
):
    test_dataset = asr_dataset.ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.test_dataset_config)
    )
    return test_dataset


def prepare_training_data_loaders(
    config: Config,
    train_dataset: asr_dataset.ASRDataset,
    eval_dataset: asr_dataset.ASRDataset,
    strategy,
    batch_size: int = None,
):
    global_batch_size = batch_size or config.learning_config.running_config.batch_size
    global_batch_size *= strategy.num_replicas_in_sync
    train_data_loader = train_dataset.create(global_batch_size)
    eval_data_loader = eval_dataset.create(global_batch_size)
    return train_data_loader, eval_data_loader, global_batch_size
