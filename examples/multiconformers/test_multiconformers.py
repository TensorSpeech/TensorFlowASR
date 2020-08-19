# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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
from tiramisu_asr.models.multiconformers import MultiConformers
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.speech_featurizers import NumpySpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer

from multiconformers_tester import MultiConformersTester
from multiconformers_dataset import MultiConformersTFRecordDataset, MultiConformersSliceDataset

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main():
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(prog="MultiConformers Training")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--saved", type=str, default=None,
                        help="Path to saved model")

    parser.add_argument("--tfrecords", type=bool, default=False,
                        help="Whether to use tfrecords")

    parser.add_argument("--nfx", type=bool, default=False,
                        help="Whether to use numpy feature extraction")

    args = parser.parse_args()

    config = UserConfig(DEFAULT_YAML, args.config, learning=True)
    lms_config = config["speech_config"]
    lms_config["feature_type"] = "log_mel_spectrogram"
    lgs_config = config["speech_config"]
    lgs_config["feature_type"] = "log_gammatone_spectrogram"

    if args.nfx:
        speech_featurizer_lms = NumpySpeechFeaturizer(lms_config)
        speech_featurizer_lgs = NumpySpeechFeaturizer(lgs_config)
    else:
        speech_featurizer_lms = TFSpeechFeaturizer(lms_config)
        speech_featurizer_lgs = TFSpeechFeaturizer(lgs_config)

    text_featurizer = TextFeaturizer(config["decoder_config"])

    tf.random.set_seed(0)
    assert args.saved

    if args.tfrecords:
        test_dataset = MultiConformersTFRecordDataset(
            config["learning_config"]["dataset_config"]["eval_paths"],
            config["learning_config"]["dataset_config"]["tfrecords_dir"],
            speech_featurizer_lms, speech_featurizer_lgs, text_featurizer,
            "test", shuffle=True
        )
    else:
        test_dataset = MultiConformersSliceDataset(
            stage="test", speech_featurizer_lms=speech_featurizer_lms,
            speech_featurizer_lgs=speech_featurizer_lgs, text_featurizer=text_featurizer,
            data_paths=config["learning_config"]["dataset_config"]["eval_paths"], shuffle=True
        )

    multiconformers = MultiConformers(
        **config["model_config"],
        vocabulary_size=text_featurizer.num_classes
    )
    multiconformers._build(speech_featurizer_lms.shape, speech_featurizer_lgs.shape)
    multiconformers.load_weights(args.saved)
    multiconformers.summary(line_length=100)

    multiconformers_tester = MultiConformersTester(
        config=config["learning_config"]["running_config"])
    multiconformers_tester.compile(multiconformers)

    multiconformers_tester.run(test_dataset)


if __name__ == "__main__":
    main()
