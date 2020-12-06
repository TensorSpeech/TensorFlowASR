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
from tensorflow_asr.utils import setup_environment, setup_devices

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Testing")

parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to saved model")

parser.add_argument("--tfrecords", default=False, action="store_true",
                    help="Whether to use tfrecords as dataset")

parser.add_argument("--mxp", default=False, action="store_true",
                    help="Enable mixed precision")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true",
                    help="Whether to only use cpu")

parser.add_argument("--subwords", type=str, default=None,
                    help="Path to file that stores generated subwords")

parser.add_argument("--output_name", type=str, default="test",
                    help="Result filename name prefix")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device], cpu=args.cpu)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRTFRecordTestDataset, ASRSliceTestDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer
from tensorflow_asr.runners.base_runners import BaseTester
from tensorflow_asr.models.conformer import Conformer

config = Config(args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    raise ValueError("subwords must be set")

tf.random.set_seed(0)
assert args.saved

if args.tfrecords:
    test_dataset = ASRTFRecordTestDataset(
        data_paths=config.learning_config.dataset_config.test_paths,
        tfrecords_dir=config.learning_config.dataset_config.tfrecords_dir,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )
else:
    test_dataset = ASRSliceTestDataset(
        data_paths=config.learning_config.dataset_config.test_paths,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )

# build model
conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
conformer._build(speech_featurizer.shape)
conformer.load_weights(args.saved, by_name=True)
conformer.summary(line_length=120)
conformer.add_featurizers(speech_featurizer, text_featurizer)

conformer_tester = BaseTester(
    config=config.learning_config.running_config,
    output_name=args.output_name
)
conformer_tester.compile(conformer)
conformer_tester.run(test_dataset)
