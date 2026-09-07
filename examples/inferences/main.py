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

"""
Transcribe one audio file with a checkpoint, through `ASRInference`.

Decoding is `ASRInference`'s job -- building the predict input, threading the states, detokenizing
-- so this file is only about getting a model and a signal in front of it. See
`tensorflow_asr/inferences.py` for the streaming form and for driving an exported `.tflite`
instead, and `examples/inferences/tflite.py` for the interpreter API underneath.
"""

import logging
import os

from tensorflow_asr import tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.inferences import ASRInference
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import cli_util, data_util, env_util, file_util, keras_util

logger = logging.getLogger(__name__)


def main(
    file_path: str,
    config_path: str,
    h5: str,
    repodir: str = os.getcwd(),
):
    """
    Parameters
    ----------
    file_path : str
        Audio to transcribe. Any format `librosa` reads; it is resampled to the rate the model
        was trained at, which `speech_config` in the config carries.
    """
    env_util.setup_seed()
    file_path = file_util.preprocess_paths(file_path)

    config = Config(config_path, training=False, repodir=repodir)
    tokenizer = tokenizers.get(config)
    tokenizer.make()

    model: BaseModel = keras_util.model_from_config(config.model_config)
    # Attached before `make()`, because the decoder turns tokens into text inside the graph.
    model.tokenizer = tokenizer
    model.make(batch_size=1)
    model.load_weights(h5, skip_mismatch=False)
    model.summary()

    signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav(file_path, sample_rate=model.feature_extraction.sample_rate))

    # `streaming=False` decodes the whole signal in one pass, which is exact for every
    # architecture. A flat signal is a batch of one, so the transcript is row 0.
    transcript = ASRInference(model=model)(signal, streaming=False)[0]
    logger.info(f"Transcript: {transcript}")


if __name__ == "__main__":
    cli_util.run(main)
