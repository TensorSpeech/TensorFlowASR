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
from tensorflow_asr.featurizers.text_featurizers import SentencePieceFeaturizer
from tensorflow_asr.utils import cli_util, env_util

logger = env_util.setup_environment()


def main(
    config_path: str,
):
    tf.keras.backend.clear_session()
    config = Config(config_path)
    SentencePieceFeaturizer.build_from_corpus(config.decoder_config)


if __name__ == "__main__":
    cli_util.run(main)
