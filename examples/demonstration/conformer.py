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

parser = argparse.ArgumentParser(prog="Conformer non streaming")

parser.add_argument("filename", metavar="FILENAME",
                    help="audio file to be played back")

parser.add_argument("--config", type=str, default=None,
                    help="Path to conformer config yaml")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to conformer saved h5 weights")

parser.add_argument("--blank", type=int, default=0,
                    help="Path to conformer tflite")

parser.add_argument("--beam_width", type=int, default=0, help="Beam width")

parser.add_argument("--num_rnns", type=int, default=1,
                    help="Number of RNN layers in prediction network")

parser.add_argument("--nstates", type=int, default=2,
                    help="Number of RNN states in prediction network (1 for GRU and 2 for LSTM)")

parser.add_argument("--statesize", type=int, default=320,
                    help="Size of RNN state in prediction network")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true",
                    help="Whether to only use cpu")

parser.add_argument("--subwords", type=str, default=None,
                    help="Path to file that stores generated subwords")

parser.add_argument("--output_name", type=str, default="test",
                    help="Result filename name prefix")

args = parser.parse_args()

setup_devices([args.device], cpu=args.cpu)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer
from tensorflow_asr.models.conformer import Conformer

config = Config(args.config, learning=False)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
if args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    text_featurizer = CharFeaturizer(config.decoder_config)
text_featurizer.decoder_config.beam_width = args.beam_width

# build model
conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
conformer._build(speech_featurizer.shape)
conformer.load_weights(args.saved, by_name=True)
conformer.summary(line_length=120)
conformer.add_featurizers(speech_featurizer, text_featurizer)

signal = read_raw_audio(args.filename)

if (args.beam_width):
    transcript = conformer.recognize_beam(signal[None, ...])
else:
    transcript = conformer.recognize(signal[None, ...])

tf.print("Transcript:", transcript[0])
