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
from model import DeepSpeech2

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs", "vivos.yml")


def main():
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(prog="Deep Speech 2 Tester")

    parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--saved_path", "-e", type=str, default=None,
                        help="Path to the model file to be exported")

    parser.add_argument("--from_weights", type=bool, default=False,
                        help="Whether to save or load only weights")

    parser.add_argument("--tfrecords", type=bool, default=False,
                        help="Whether to use tfrecords dataset")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for testing")

    args = parser.parse_args()

    tf.random.set_seed(0)
    assert args.export

    config = UserConfig(DEFAULT_YAML, args.config, learning=True)
    speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
    text_featurizer = TextFeaturizer(config["decoder_config"])
    # Build DS2 model
    f, c = speech_featurizer.compute_feature_dim()
    ds2_model = DeepSpeech2(input_shape=[None, f, c],
                            arch_config=config["model_config"],
                            num_classes=text_featurizer.num_classes,
                            name="deepspeech2")
    ds2_model._build([1, 50, f, c])
    ds2_model.summary(line_length=100)

    if args.tfrecords:
        test_dataset = ASRTFRecordDataset(
            config["learning_config"]["dataset_config"]["test_paths"],
            config["learning_config"]["dataset_config"]["tfrecords_dir"],
            speech_featurizer, text_featurizer, "test",
            augmentations=config["learning_config"]["augmentations"], shuffle=False
        ).create(args.batch_size)
    else:
        test_dataset = ASRSliceDataset(
            stage="test", speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            data_paths=config["learning_config"]["dataset_config"]["eval_paths"],
            shuffle=False
        ).create(args.batch_size)

    ctc_tester = BaseTester(
        config=config["learning_config"]["running_config"],
        saved_path=args.saved_path, from_weights=args.from_weights
    )
    ctc_tester.compile(ds2_model, speech_featurizer, text_featurizer)
    ctc_tester.run(test_dataset)


if __name__ == "__main__":
    main()
