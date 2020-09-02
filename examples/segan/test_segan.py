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

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

parser = argparse.ArgumentParser(prog="SEGAN")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to saved model")

parser.add_argument("--mxp", default=False, action="store_true",
                    help="Enable mixed precision")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device])

from tiramisu_asr.runners.segan_runners import SeganTester
from tiramisu_asr.datasets.segan_dataset import SeganAugTestDataset
from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.models.segan import Generator

config = UserConfig(DEFAULT_YAML, args.config, learning=True)

tf.random.set_seed(0)
assert args.saved

dataset = SeganAugTestDataset(
    clean_dir=config["learning_config"]["dataset_config"]["test_paths"],
    noises_config=config["learning_config"]["dataset_config"]["noise_config"],
    speech_config=config["speech_config"]
)

generator = Generator(
    window_size=config["speech_config"]["window_size"],
    **config["model_config"]
)
generator._build()
generator.load_weights(args.saved, by_name=True)
generator.summary(line_length=150)

segan_tester = SeganTester(config["speech_config"],
                           config["learning_config"]["running_config"])

segan_tester.compile(generator)
segan_tester.run(dataset)
