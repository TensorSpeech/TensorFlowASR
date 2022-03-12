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

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio


def main(
    saved_model: str = None,
    filename: str = None,
):
    tf.keras.backend.clear_session()

    module = tf.saved_model.load(export_dir=saved_model)

    signal = read_raw_audio(filename)
    transcript = module.pred(signal)

    print("Transcript: ", "".join([chr(u) for u in transcript]))


if __name__ == "__main__":
    fire.Fire(main)
