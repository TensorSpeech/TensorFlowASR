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
from tensorflow_asr.utils import env_util, file_util

logger = env_util.setup_environment()
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer, CharFeaturizer
from tensorflow_asr.models.ctc.deepspeech2 import DeepSpeech2

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="DeepSpeech2 TFLite")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None, help="Path to saved model")

parser.add_argument("--subwords", type=str, default=None, help="Use subwords")

parser.add_argument("output", type=str, default=None, help="TFLite file path to be exported")

args = parser.parse_args()

assert args.saved and args.output

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.subwords:
    text_featurizer = SubwordFeaturizer(config.decoder_config)
else:
    text_featurizer = CharFeaturizer(config.decoder_config)

# build model
deepspeech2 = DeepSpeech2(**config.model_config, vocabulary_size=text_featurizer.num_classes)
deepspeech2.make(speech_featurizer.shape)
deepspeech2.load_weights(args.saved, by_name=True)
deepspeech2.summary(line_length=100)
deepspeech2.add_featurizers(speech_featurizer, text_featurizer)

concrete_func = deepspeech2.make_tflite_function().get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

args.output = file_util.preprocess_paths(args.output)
with open(args.output, "wb") as tflite_out:
    tflite_out.write(tflite_model)
