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

import logging
import os

from tensorflow_asr import keras, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import app_util, cli_util, env_util, keras_util

logger = logging.getLogger(__name__)


def main(
    config_path: str,
    output: str,
    h5: str = None,
    lm_h5: str = None,
    internal_lm_h5: str = None,
    bs: int = 1,
    beam_width: int = 0,
    nchunks: int = 1,
    repodir: str = os.getcwd(),
):
    """
    Convert a checkpoint to a TFLite flatbuffer.

    Parameters
    ----------
    beam_width : int
        Hypotheses per utterance. `0` exports the greedy decoder, anything positive exports the
        ALSD++ beam search together with the language model settings from `decoder_config` --
        this flag overrides `decoder_config.beam_width`, which is 0 in every shipped config.
    lm_h5 : str
        Weights of the external language model fused into the exported beam search, from
        `tensorflow_asr train_lm --target=external`. Same flag as `tensorflow_asr test` for the
        same reason: which trained copy you ship is a property of the run, not of the config.
        Only read when `lm_config.external_config` describes a model; without it that model is
        frozen into the flatbuffer with its *initial* weights, which is never what you want.
    internal_lm_h5 : str
        Same, for the low-order language model LODR subtracts (`train_lm --target=internal`,
        `lm_config.internal_config`). Only read when `decoder_config.lm_type` is "lodr".
    nchunks : int
        Attention chunks a streaming client should feed per call, recorded in the exported model's
        metadata. The graph is unaffected -- this only changes the chunk geometry a client reads
        back out of the flatbuffer, and a client can rescale it without re-exporting. See
        `BaseModel.get_tflite_metadata`.
    """
    assert output
    keras.backend.clear_session()
    env_util.setup_seed()

    config = Config(config_path, training=False, repodir=repodir)
    tokenizer = tokenizers.get(config)
    tokenizer.make()

    logger.info(f"Configs: {str(config)}")

    model: BaseModel = keras_util.model_from_config(config.model_config)
    model.tokenizer = tokenizer
    model.make(batch_size=bs)
    if h5 and tf.io.gfile.exists(h5):
        model.load_weights(h5, skip_mismatch=False)
    model.summary()
    # After `make()`, which `make_lm` requires, and only meaningful for a beam export -- the greedy
    # decoder never reads a language model. Validated the same way `scripts/test.py` validates it,
    # so a combination that would silently decode without the correction asked for is caught here
    # rather than after the conversion has run.
    model.make_lm(config.lm_config, lm_weights=lm_h5, internal_lm_weights=internal_lm_h5)  # no-op unless lm_config sets a model
    if beam_width > 0:
        app_util.validate_lm(model, config.decoder_config, lm_h5=lm_h5, internal_lm_h5=internal_lm_h5, beam_width=beam_width)

    app_util.convert_tflite(model=model, output=output, batch_size=bs, beam_width=beam_width, nchunks=nchunks)


if __name__ == "__main__":
    cli_util.run(main)
