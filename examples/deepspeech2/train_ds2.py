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

import os
import argparse
from tiramisu_asr.utils import setup_environment, setup_strategy

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs", "vivos.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Deep Speech 2 Training")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10,
                    help="Max number of checkpoints to keep")

parser.add_argument("--eval_train_ratio", type=int, default=1,
                    help="ratio between train batch size and eval batch size")

parser.add_argument("--tfrecords", type=bool, default=False,
                    help="Whether to use tfrecords dataset")

parser.add_argument("--devices", type=int, nargs="*", default=[0],
                    help="Devices' ids to apply distributed training")

args = parser.parse_args()

strategy = setup_strategy(args.devices)

from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.runners.ctc_runners import CTCTrainer
from model import DeepSpeech2

config = UserConfig(DEFAULT_YAML, args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
text_featurizer = TextFeaturizer(config["decoder_config"])

tf.random.set_seed(2020)

if args.tfrecords:
    train_dataset = ASRTFRecordDataset(
        config["learning_config"]["dataset_config"]["train_paths"],
        config["learning_config"]["dataset_config"]["tfrecords_dir"],
        speech_featurizer, text_featurizer, "train",
        augmentations=config["learning_config"]["augmentations"], shuffle=True,
    )
    eval_dataset = ASRTFRecordDataset(
        config["learning_config"]["dataset_config"]["eval_paths"],
        config["learning_config"]["dataset_config"]["tfrecords_dir"],
        speech_featurizer, text_featurizer, "eval", shuffle=False
    )
else:
    train_dataset = ASRSliceDataset(
        stage="train", speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        data_paths=config["learning_config"]["dataset_config"]["eval_paths"],
        augmentations=config["learning_config"]["augmentations"], shuffle=True
    )
    eval_dataset = ASRSliceDataset(
        stage="train", speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        data_paths=config["learning_config"]["dataset_config"]["eval_paths"],
        shuffle=True
    )

ctc_trainer = CTCTrainer(text_featurizer, config["learning_config"]["running_config"])
# Build DS2 model
with ctc_trainer.strategy.scope():
    ds2_model = DeepSpeech2(input_shape=speech_featurizer.shape,
                            arch_config=config["model_config"],
                            num_classes=text_featurizer.num_classes,
                            name="deepspeech2")
    ds2_model._build(speech_featurizer.shape)
    ds2_model.summary(line_length=150)
# Compile
ctc_trainer.compile(ds2_model, config["learning_config"]["optimizer_config"],
                    max_to_keep=args.max_ckpts)

ctc_trainer.fit(train_dataset, eval_dataset, args.eval_train_ratio)
