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

parser = argparse.ArgumentParser(prog="Conformer Testing")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--h5", type=str, default=None, help="Path to saved h5 weights")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--subwords", default=False, action="store_true", help="Use subwords")

parser.add_argument("--output_dir", type=str, default=None, help="Output directory for saved model")

args = parser.parse_args()

assert args.h5
assert args.output_dir

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SentencePieceFeaturizer, SubwordFeaturizer
from tensorflow_asr.models.transducer.conformer import Conformer

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    logger.info("Use SentencePiece ...")
    text_featurizer = SentencePieceFeaturizer(config.decoder_config)
elif args.subwords:
    logger.info("Use subwords ...")
    text_featurizer = SubwordFeaturizer(config.decoder_config)
else:
    logger.info("Use characters ...")
    text_featurizer = CharFeaturizer(config.decoder_config)

tf.random.set_seed(0)

# build model
conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
conformer.make(speech_featurizer.shape)
conformer.load_weights(args.h5, by_name=True)
conformer.summary(line_length=100)
conformer.add_featurizers(speech_featurizer, text_featurizer)


class ConformerModule(tf.Module):
    def __init__(self, model: Conformer, name=None):
        super().__init__(name=name)
        self.model = model
        self.num_rnns = config.model_config["prediction_num_rnns"]
        self.rnn_units = config.model_config["prediction_rnn_units"]
        self.rnn_nstates = 2 if config.model_config["prediction_rnn_type"] == "lstm" else 1

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def pred(self, signal):
        predicted = tf.constant(0, dtype=tf.int32)
        states = tf.zeros([self.num_rnns, self.rnn_nstates, 1, self.rnn_units], dtype=tf.float32)
        features = self.model.speech_featurizer.tf_extract(signal)
        encoded = self.model.encoder_inference(features)
        hypothesis = self.model._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=False)
        transcript = self.model.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript


module = ConformerModule(model=conformer)
tf.saved_model.save(module, export_dir=args.output_dir, signatures=module.pred.get_concrete_function())
