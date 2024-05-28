# Copyright 2024 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import keras, tf, tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import cli_util, env_util, file_util

env_util.setup_logging()


def main(
    config_path: str,
    output: str,
    h5: str = None,
    bs: int = 2,
    save_format: str = "h5",
    repodir: str = os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
):
    assert output
    keras.backend.clear_session()
    env_util.setup_seed()

    config = Config(config_path, training=False, repodir=repodir)
    tokenizer = tokenizers.get(config)

    model: BaseModel = keras.models.model_from_config(config.model_config)
    model.tokenizer = tokenizer
    model.make(batch_size=bs)
    if h5 and tf.io.gfile.exists(h5):
        model.load_weights(h5, by_name=file_util.is_hdf5_filepath(h5))
    model.summary()

    model.save(output, save_format=save_format)
    loaded_model: BaseModel = keras.models.load_model(output)
    print(loaded_model.to_json())
    loaded_model.summary()


if __name__ == "__main__":
    cli_util.run(main)
