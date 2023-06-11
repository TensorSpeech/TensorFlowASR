# Copyright 2022 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import tf
from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRDataset
from tensorflow_asr.featurizers import text_featurizers
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.utils import cli_util

logger = tf.get_logger()


def main(
    config_path: str,
):
    config = Config(config_path)
    if not config.decoder_config.vocabulary:
        raise ValueError("decoder_config.vocabulary must be defined")
    metadata = f"{os.path.splitext(config.decoder_config.vocabulary)[0]}.metadata.json"

    logger.info("Preparing vocab ...")
    if config.decoder_config.type == "sentencepiece":
        text_featurizers.SentencePieceFeaturizer.build_from_corpus(config.decoder_config)
    elif config.decoder_config.type == "wordpiece":
        text_featurizers.WordPieceFeaturizer.build_from_corpus(config.decoder_config)

    logger.info("Preparing train metadata ...")
    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)
    train_dataset = ASRDataset(
        data_paths=config.decoder_config.train_files,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="train",
        shuffle=False,
    )
    train_dataset.update_metadata(metadata)

    logger.info("Preparing eval metadata ...")
    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)
    eval_dataset = ASRDataset(
        data_paths=config.decoder_config.eval_files,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="eval",
        shuffle=False,
    )
    eval_dataset.update_metadata(metadata)


if __name__ == "__main__":
    cli_util.run(main)
