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
from tiramisu_asr.utils import setup_environment

setup_environment()
import tensorflow as tf

from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.runners.base_runners import BaseTester
from tiramisu_asr.models.conformer import Conformer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main():
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(prog="Conformer Testing")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--saved", type=str, default=None,
                        help="Path to saved model")

    parser.add_argument("--tfrecords", type=bool, default=False,
                        help="Whether to use tfrecords as dataset")

    args = parser.parse_args()

    config = UserConfig(DEFAULT_YAML, args.config, learning=True)
    speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
    text_featurizer = TextFeaturizer(config["decoder_config"])

    tf.random.set_seed(0)
    assert args.saved

    if args.tfrecords:
        test_dataset = ASRTFRecordDataset(
            config["learning_config"]["dataset_config"]["test_paths"],
            config["learning_config"]["dataset_config"]["tfrecords_dir"],
            speech_featurizer, text_featurizer, "test",
            augmentations=config["learning_config"]["augmentations"], shuffle=False
        )
    else:
        test_dataset = ASRSliceDataset(
            stage="test", speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            data_paths=config["learning_config"]["dataset_config"]["test_paths"],
            shuffle=False
        )

    # build model
    conformer = Conformer(
        vocabulary_size=text_featurizer.num_classes,
        **config["model_config"]
    )
    conformer._build(speech_featurizer.shape)
    conformer.summary(line_length=150)
    conformer.load_weights(args.saved)
    conformer.add_featurizers(speech_featurizer, text_featurizer)

    conformer_tester = BaseTester(config=config["learning_config"]["running_config"])
    conformer_tester.compile(conformer)
    conformer_tester.run(test_dataset)


if __name__ == "__main__":
    main()
