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

from tensorflow_asr import tf, tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.datasets import ASRDataset
from tensorflow_asr.utils import cli_util, env_util

env_util.setup_logging()
logger = tf.get_logger()


def main(
    config_path: str,
    datadir: str,
    repodir: str = os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
):
    config = Config(config_path, repodir=repodir, datadir=datadir)
    if not config.decoder_config.vocabulary:
        raise ValueError("decoder_config.vocabulary must be defined")

    logger.info("Preparing vocab ...")
    tokenizers.build(config=config)
    tokenizer = tokenizers.get(config=config)

    logger.info("Preparing train metadata ...")
    config.data_config.train_dataset_config.drop_remainder = False
    config.data_config.train_dataset_config.shuffle = False
    train_dataset = ASRDataset(tokenizer=tokenizer, **vars(config.data_config.train_dataset_config))
    train_dataset.update_metadata()

    logger.info("Preparing eval metadata ...")
    config.data_config.eval_dataset_config.drop_remainder = False
    config.data_config.eval_dataset_config.shuffle = False
    eval_dataset = ASRDataset(tokenizer=tokenizer, **vars(config.data_config.eval_dataset_config))
    eval_dataset.update_metadata()


if __name__ == "__main__":
    cli_util.run(main)
