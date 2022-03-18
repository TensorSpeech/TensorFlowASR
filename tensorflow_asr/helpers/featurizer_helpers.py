# Copyright 2022 Huy Le Nguyen (@usimarit)
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

import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers

logger = tf.get_logger()


def prepare_featurizers(
    config: Config,
):
    speech_featurizer = speech_featurizers.SpeechFeaturizer(config.speech_config)
    if config.decoder_config.type == "sentencepiece":
        logger.info("Loading SentencePiece model ...")
        text_featurizer = text_featurizers.SentencePieceFeaturizer(config.decoder_config)
    elif config.decoder_config.type == "subwords":
        logger.info("Loading subwords ...")
        text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
    elif config.decoder_config.type == "wordpiece":
        logger.info("Loading wordpiece ...")
        text_featurizer = text_featurizers.WordPieceFeaturizer(config.decoder_config)
    elif config.decoder_config.type == "characters":
        logger.info("Use characters ...")
        text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)
    else:
        raise ValueError(f"type must be in {text_featurizers.TEXT_FEATURIZER_TYPES}, received {config.decoder_config.type}")
    return speech_featurizer, text_featurizer
