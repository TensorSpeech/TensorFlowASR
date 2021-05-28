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
from tensorflow_asr.utils import env_util, math_util

logger = env_util.setup_environment()
import tensorflow as tf

parser = argparse.ArgumentParser(prog="Conformer non streaming")

parser.add_argument("filename", metavar="FILENAME", help="audio file to be played back")

parser.add_argument("--config", type=str, default=None, help="Path to conformer config yaml")

parser.add_argument("--saved", type=str, default=None, help="Path to conformer saved h5 weights")

parser.add_argument("--beam_width", type=int, default=0, help="Beam width")

parser.add_argument("--timestamp", default=False, action="store_true", help="Return with timestamp")

parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true", help="Whether to only use cpu")

parser.add_argument("--subwords", type=str, default=None, help="Path to file that stores generated subwords")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

args = parser.parse_args()

env_util.setup_devices([args.device], cpu=args.cpu)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer, SentencePieceFeaturizer
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.utils.data_util import create_inputs

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
if args.sentence_piece:
    logger.info("Loading SentencePiece model ...")
    text_featurizer = SentencePieceFeaturizer.load_from_file(config.decoder_config, args.subwords)
elif args.subwords and os.path.exists(args.subwords):
    logger.info("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    text_featurizer = CharFeaturizer(config.decoder_config)
text_featurizer.decoder_config.beam_width = args.beam_width

# build model
conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
conformer.make(speech_featurizer.shape)
conformer.load_weights(args.saved, by_name=True, skip_mismatch=True)
conformer.summary(line_length=120)
conformer.add_featurizers(speech_featurizer, text_featurizer)

signal = read_raw_audio(args.filename)
features = speech_featurizer.tf_extract(signal)
input_length = tf.shape(features)[0]

if args.beam_width:
    inputs = create_inputs(features[None, ...], input_length[None, ...])
    transcript = conformer.recognize_beam(inputs)
    logger.info(f"Transcript: {transcript[0].numpy().decode('UTF-8')}")
elif args.timestamp:
    transcript, stime, etime, _, _ = conformer.recognize_tflite_with_timestamp(
        signal, tf.constant(text_featurizer.blank, dtype=tf.int32), conformer.predict_net.get_initial_state())
    logger.info(f"Transcript: {transcript}")
    logger.info(f"Start time: {stime}")
    logger.info(f"End time: {etime}")
else:
    code_points, _, _ = conformer.recognize_tflite(
        signal, tf.constant(text_featurizer.blank, dtype=tf.int32), conformer.predict_net.get_initial_state())
    transcript = tf.strings.unicode_encode(code_points, 'UTF-8').numpy().decode('UTF-8')
    logger.info(f"Transcript: {transcript}")
