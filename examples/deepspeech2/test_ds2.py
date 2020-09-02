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
from tiramisu_asr.utils import setup_environment, setup_devices

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs", "vivos.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Deep Speech 2 Tester")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to the model file to be exported")

parser.add_argument("--tfrecords", default=False, action="store_true",
                    help="Whether to use tfrecords dataset")

parser.add_argument("--mxp", default=False, action="store_true",
                    help="Enable mixed precision")

parser.add_argument("--bs", type=int, default=None, help="Batch size")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device])

from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.runners.base_runners import BaseTester
from model import DeepSpeech2

tf.random.set_seed(0)
assert args.export

config = UserConfig(DEFAULT_YAML, args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
text_featurizer = TextFeaturizer(config["decoder_config"])
# Build DS2 model
ds2_model = DeepSpeech2(input_shape=speech_featurizer.shape,
                        arch_config=config["model_config"],
                        num_classes=text_featurizer.num_classes,
                        name="deepspeech2")
ds2_model._build(speech_featurizer.shape)
ds2_model.load_weights(args.saved, by_name=True)
ds2_model.summary(line_length=150)
ds2_model.add_featurizers(speech_featurizer, text_featurizer)

if args.tfrecords:
    test_dataset = ASRTFRecordDataset(
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        tfrecords_dir=config["learning_config"]["dataset_config"]["tfrecords_dir"],
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )
else:
    test_dataset = ASRSliceDataset(
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )

ctc_tester = BaseTester(config=config["learning_config"]["running_config"])
ctc_tester.compile(ds2_model)
ctc_tester.run(test_dataset, batch_size=args.bs)
