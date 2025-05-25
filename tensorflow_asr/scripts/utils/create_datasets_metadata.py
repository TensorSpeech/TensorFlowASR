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


import logging
import os

from tensorflow_asr import datasets, tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.utils import cli_util

logger = logging.getLogger(__name__)


def main(
    config_path: str,
    datadir: str,
    dataset_type: str,
    repodir: str = os.getcwd(),
):
    config = Config(config_path, repodir=repodir, datadir=datadir)
    if not config.decoder_config.vocabulary:
        raise ValueError("decoder_config.vocabulary must be defined")

    tokenizer = tokenizers.get(config)

    logger.info("Preparing train metadata ...")
    config.data_config.train_dataset_config.drop_remainder = False
    config.data_config.train_dataset_config.shuffle = False
    train_dataset = datasets.get(
        tokenizer=tokenizer,
        dataset_config=config.data_config.train_dataset_config,
        dataset_type=dataset_type,
    )
    tokenizer.build(train_dataset)
    tokenizer.make()
    train_dataset.update_metadata()

    logger.info("Preparing eval metadata ...")
    config.data_config.eval_dataset_config.drop_remainder = False
    config.data_config.eval_dataset_config.shuffle = False
    eval_dataset = datasets.get(
        tokenizer=tokenizer,
        dataset_config=config.data_config.eval_dataset_config,
        dataset_type=dataset_type,
    )
    eval_dataset.update_metadata()


if __name__ == "__main__":
    cli_util.run(main)
