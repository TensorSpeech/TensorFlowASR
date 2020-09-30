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
from tiramisu_asr.utils import setup_environment, setup_devices

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="MultiConformers Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to saved model")

parser.add_argument("--tfrecords", default=False, action="store_true",
                    help="Whether to use tfrecords")

parser.add_argument("--mxp", default=False, action="store_true",
                    help="Enable mixed precision")

parser.add_argument("--nfx", default=False, action="store_true",
                    help="Whether to use numpy feature extraction")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true",
                    help="Whether to only use cpu")

parser.add_argument("--output_name", type=str, default="test",
                    help="Result filename name prefix")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device], cpu=args.cpu)

from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.models.multiconformers import MultiConformers
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.speech_featurizers import NumpySpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import CharFeaturizer

from multiconformers_tester import MultiConformersTester
from multiconformers_dataset import MultiConformersTFRecordDataset, MultiConformersSliceDataset

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

text_featurizer = CharFeaturizer(config["decoder_config"])

tf.random.set_seed(0)
assert args.saved

if args.tfrecords:
    test_dataset = MultiConformersTFRecordDataset(
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        tfrecords_dir=config["learning_config"]["dataset_config"]["tfrecords_dir"],
        speech_featurizer_lms=speech_featurizer_lms,
        speech_featurizer_lgs=speech_featurizer_lgs,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )
else:
    test_dataset = MultiConformersSliceDataset(
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        speech_featurizer_lms=speech_featurizer_lms,
        speech_featurizer_lgs=speech_featurizer_lgs,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )

multiconformers = MultiConformers(
    **config["model_config"],
    vocabulary_size=text_featurizer.num_classes
)
multiconformers._build(speech_featurizer_lms.shape, speech_featurizer_lgs.shape)
multiconformers.load_weights(args.saved, by_name=True)
multiconformers.summary(line_length=120)
multiconformers.add_featurizers(speech_featurizer_lms, speech_featurizer_lgs, text_featurizer)

multiconformers_tester = MultiConformersTester(
    config=config["learning_config"]["running_config"],
    output_name=args.output_name
)
multiconformers_tester.compile(multiconformers)

multiconformers_tester.run(test_dataset)
