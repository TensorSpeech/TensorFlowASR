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
from tensorflow_asr.utils import setup_environment

setup_environment()
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer
from tensorflow_asr.models.contextnet import ContextNet

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="ContextNet Testing")

parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to saved model")

parser.add_argument("--subwords", type=str, default=None,
                    help="Path to file that stores generated subwords")

parser.add_argument("output", type=str, default=None,
                    help="TFLite file path to be exported")

args = parser.parse_args()

assert args.saved and args.output

config = Config(args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    raise ValueError("subwords must be set")

# build model
contextnet = ContextNet(**config.model_config, vocabulary_size=text_featurizer.num_classes)
contextnet._build(speech_featurizer.shape)
contextnet.load_weights(args.saved, by_name=True)
contextnet.summary(line_length=150)
contextnet.add_featurizers(speech_featurizer, text_featurizer)

concrete_func = contextnet.make_tflite_function(greedy=True).get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

if not os.path.exists(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))
with open(args.output, "wb") as tflite_out:
    tflite_out.write(tflite_model)
