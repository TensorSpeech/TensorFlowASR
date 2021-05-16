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
from tensorflow_asr.utils import env_util, math_util, data_util

logger = env_util.setup_environment()
import tensorflow as tf

parser = argparse.ArgumentParser(prog="Rnn Transducer non streaming")

parser.add_argument("filename", metavar="FILENAME", help="audio file to be played back")

parser.add_argument("--config", type=str, default=None, help="Path to rnnt config yaml")

parser.add_argument("--saved", type=str, default=None, help="Path to rnnt saved h5 weights")

parser.add_argument("--beam_width", type=int, default=0, help="Beam width")

parser.add_argument("--timestamp", default=False, action="store_true", help="Return with timestamp")

parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true", help="Whether to only use cpu")

parser.add_argument("--subwords", default=False, action="store_true", help="Path to file that stores generated subwords")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

args = parser.parse_args()

env_util.setup_devices([args.device], cpu=args.cpu)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer, SentencePieceFeaturizer
from tensorflow_asr.models.transducer.rnn_transducer import RnnTransducer

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
if args.sentence_piece:
    logger.info("Loading SentencePiece model ...")
    text_featurizer = SentencePieceFeaturizer(config.decoder_config)
elif args.subwords:
    logger.info("Loading subwords ...")
    text_featurizer = SubwordFeaturizer(config.decoder_config)
else:
    text_featurizer = CharFeaturizer(config.decoder_config)
text_featurizer.decoder_config.beam_width = args.beam_width

# build model
rnnt = RnnTransducer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
rnnt.make(speech_featurizer.shape)
rnnt.load_weights(args.saved, by_name=True, skip_mismatch=True)
rnnt.summary(line_length=120)
rnnt.add_featurizers(speech_featurizer, text_featurizer)

signal = read_raw_audio(args.filename)
features = speech_featurizer.tf_extract(signal)
input_length = math_util.get_reduced_length(tf.shape(features)[0], rnnt.time_reduction_factor)

if args.beam_width:
    transcript = rnnt.recognize_beam(
        data_util.create_inputs(
            inputs=features[None, ...],
            inputs_length=input_length[None, ...]
        )
    )
    logger.info("Transcript:", transcript[0].numpy().decode("UTF-8"))
elif args.timestamp:
    transcript, stime, etime, _, _, _ = rnnt.recognize_tflite_with_timestamp(
        signal=signal,
        predicted=tf.constant(text_featurizer.blank, dtype=tf.int32),
        encoder_states=rnnt.encoder.get_initial_state(),
        prediction_states=rnnt.predict_net.get_initial_state()
    )
    logger.info("Transcript:", transcript)
    logger.info("Start time:", stime)
    logger.info("End time:", etime)
else:
    transcript = rnnt.recognize(
        data_util.create_inputs(
            inputs=features[None, ...],
            inputs_length=input_length[None, ...]
        )
    )
    logger.info("Transcript:", transcript[0].numpy().decode("UTF-8"))
