# Copyright 2020 Huy Le Nguyen (@nglehuy)
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
from typing import List

from tensorflow_asr import datasets, tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.utils import cli_util


def main(
    config_path: str,
    datadir: str,
    modes: List[str],
    repodir: str = os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
):
    config = Config(config_path, repodir=repodir, datadir=datadir)
    tokenizer = tokenizers.get(config=config)
    for mode in modes:
        dat = datasets.get(
            tokenizer=tokenizer,
            dataset_config=getattr(config.data_config, f"{mode}_dataset_config"),
            dataset_type="tfrecord",
        )
        dat.create_tfrecords()


if __name__ == "__main__":
    cli_util.run(main)
