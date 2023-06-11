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

from tensorflow_asr import tf
from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.text_featurizers import WordPieceFeaturizer
from tensorflow_asr.utils import cli_util, env_util

logger = tf.get_logger()


def main(
    config_path: str,
):
    config = Config(config_path)
    WordPieceFeaturizer.build_from_corpus(decoder_config=config.decoder_config)


if __name__ == "__main__":
    cli_util.run(main)
