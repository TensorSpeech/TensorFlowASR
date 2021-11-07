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

import argparse
import os

from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

parser.add_argument("--saved_model", type=str, default=None, help="The file path of saved model")

parser.add_argument("filename", type=str, default=None, help="Audio file path")

args = parser.parse_args()

from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio

module = tf.saved_model.load(export_dir=args.saved_model)

signal = read_raw_audio(args.filename)
transcript = module.pred(signal)

print("Transcript: ", "".join([chr(u) for u in transcript]))
