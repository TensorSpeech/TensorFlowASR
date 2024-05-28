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

from tensorflow_asr import keras, schemas, tf, tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.models import base_model
from tensorflow_asr.utils import cli_util, data_util, env_util, file_util

env_util.setup_logging()
logger = tf.get_logger()


def main(
    file_path: str,
    config_path: str,
    h5: str,
    repodir: str = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")),
):
    env_util.setup_seed()
    file_path = file_util.preprocess_paths(file_path)

    config = Config(config_path, training=False, repodir=repodir)
    tokenizer = tokenizers.get(config)

    model: base_model.BaseModel = keras.models.model_from_config(config.model_config)
    model.make(batch_size=1)
    model.load_weights(h5, by_name=file_util.is_hdf5_filepath(h5), skip_mismatch=False)
    model.summary()

    signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav(file_path))
    signal = tf.reshape(signal, [1, -1])
    signal_length = tf.reshape(tf.shape(signal)[1], [1])

    outputs = model.recognize(
        schemas.PredictInput(
            inputs=signal,
            inputs_length=signal_length,
            previous_tokens=model.get_initial_tokens(),
            previous_encoder_states=model.get_initial_encoder_states(),
            previous_decoder_states=model.get_initial_decoder_states(),
        )
    )
    transcript = tokenizer.detokenize(outputs.tokens)[0].numpy().decode("utf-8")
    logger.info(f"Transcript: {transcript}")


if __name__ == "__main__":
    cli_util.run(main)
