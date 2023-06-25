# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import tf  # import to aid logging messages
from tensorflow_asr.config import Config
from tensorflow_asr.featurizers import text_featurizers
from tensorflow_asr.utils import app_util, cli_util, env_util, file_util


def main(
    config_path: str,
    h5: str = None,
    output: str = None,
    repodir: str = os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
):
    assert h5 and output
    tf.keras.backend.clear_session()
    env_util.setup_seed()
    tf.compat.v1.enable_control_flow_v2()

    config = Config(config_path, training=False, repodir=repodir)
    text_featurizer = text_featurizers.get(config)

    model = tf.keras.models.model_from_config(config.model_config)
    model.make()
    model.load_weights(h5, by_name=file_util.is_hdf5_filepath(h5))
    model.summary()
    model.text_featurizer = text_featurizer

    app_util.convert_tflite(model=model, output=output)


if __name__ == "__main__":
    cli_util.run(main)
