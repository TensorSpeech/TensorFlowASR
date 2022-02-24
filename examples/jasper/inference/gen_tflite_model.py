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

import os
import fire
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import exec_helpers, featurizer_helpers
from tensorflow_asr.models.ctc.jasper import Jasper

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main(
    config: str = DEFAULT_YAML,
    h5: str = None,
    subwords: bool = False,
    sentence_piece: bool = False,
    output: str = None,
):
    assert h5 and output
    tf.keras.backend.clear_session()
    tf.compat.v1.enable_control_flow_v2()

    config = Config(config)
    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    jasper = Jasper(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    jasper.make(speech_featurizer.shape)
    jasper.load_weights(h5, by_name=True)
    jasper.summary(line_length=100)
    jasper.add_featurizers(speech_featurizer, text_featurizer)

    exec_helpers.convert_tflite(model=jasper, output=output)


if __name__ == "__main__":
    fire.Fire(main)
