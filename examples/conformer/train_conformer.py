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
import math
import argparse
from tiramisu_asr.utils import setup_environment

setup_environment()
import tensorflow as tf

from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.runners.transducer_runners import TransducerTrainer
from tiramisu_asr.models.conformer import Conformer
from tiramisu_asr.optimizers.schedules import TransformerSchedule

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main():
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(prog="Conformer Training")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--max_ckpts", type=int, default=10,
                        help="Max number of checkpoints to keep")

    parser.add_argument("--tfrecords", type=bool, default=False,
                        help="Whether to use tfrecords")

    parser.add_argument("--eval_train_ratio", type=int, default=1,
                        help="ratio between train batch size and eval batch size")

    args = parser.parse_args()

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
            data_paths=config["learning_config"]["dataset_config"]["train_paths"],
            augmentations=config["learning_config"]["augmentations"], shuffle=True,
        )

        eval_dataset = ASRSliceDataset(
            stage="eval", speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            data_paths=config["learning_config"]["dataset_config"]["eval_paths"],
            shuffle=False
        )

    conformer_trainer = TransducerTrainer(
        config=config["learning_config"]["running_config"],
        text_featurizer=text_featurizer
    )

    with conformer_trainer.strategy.scope():
        # build model
        conformer = Conformer(
            **config["model_config"],
            vocabulary_size=text_featurizer.num_classes
        )
        conformer._build(speech_featurizer.shape)
        conformer.summary(line_length=150)

        optimizer_config = config["learning_config"]["optimizer_config"]
        optimizer = tf.keras.optimizers.Adam(
            TransformerSchedule(
                d_model=config["model_config"]["dmodel"],
                warmup_steps=optimizer_config["warmup_steps"],
                max_lr=(0.05 / math.sqrt(config["model_config"]["dmodel"]))
            ),
            beta_1=optimizer_config["beta1"],
            beta_2=optimizer_config["beta2"],
            epsilon=optimizer_config["epsilon"]
        )

    conformer_trainer.compile(model=conformer, optimizer=optimizer,
                              max_to_keep=args.max_ckpts)

    conformer_trainer.fit(train_dataset, eval_dataset, args.eval_train_ratio)


if __name__ == "__main__":
    main()