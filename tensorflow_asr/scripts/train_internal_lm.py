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
"""Fit the internal language model -- the low-order n-gram LODR subtracts"""

import logging
import os

import numpy as np

from tensorflow_asr import datasets, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel, save_lm
from tensorflow_asr.utils import cli_util

logger = logging.getLogger(__name__)


def main(
    config_path: str,
    datadir: str,
    modeldir: str,
    repodir: str = os.getcwd(),
    **kwargs,
):
    """
    Fit `lm_config.internal_config` by counting, and write it to `<modeldir>/lm`.

    The internal LM is the one **subtracted**, weighted by `decoder_config.lm_beta`, when
    `decoder_config.lm_type` is "lodr". A transducer trained on paired speech and text picks up a
    language model of its training transcripts whether or not anyone asked for one, and that
    implicit model fights the external LM on any domain it was not trained on. LODR approximates it
    with a cheap low-order n-gram so it can be removed.

    **Fit this on the ASR training transcripts**, and nothing else. What it approximates is the
    internal LM the transducer picked up from exactly that text, so point
    `data_config.lm_dataset_config.data_paths` at the transcript `.tsv` files. Fitting it on
    target-domain text instead would make the correction subtract the very knowledge fusion is
    adding.

    There is no gradient descent here and no flags for it. A `BigramLanguageModel`'s maximum
    likelihood estimate is a ratio of counts, exact and available in a single pass, so `--epochs`,
    `--bs` and `--learning-rate` would have nothing to do. Counting runs on the host, so there is no
    device or strategy setup either.

    Parameters
    ----------
    config_path : str
        The same config the ASR model uses, so the tokenizer -- and therefore the vocabulary the LM
        is indexed against -- matches.
    datadir : str
    modeldir : str
        Weights are written to `<modeldir>/lm/internal.weights.h5`. Pass that to
        `tensorflow_asr test --internal-lm-h5`.
    """
    config = Config(config_path, training=False, repodir=repodir, datadir=datadir, modeldir=modeldir, **kwargs)
    model_config = config.lm_config.internal_config
    if not model_config or not model_config.get("class_name"):
        raise ValueError(f"`lm_config.internal_config` is not set in {config_path}, nothing to fit")

    tokenizer = tokenizers.get(config)
    tokenizer.make()
    logger.info(f"Vocabulary size {tokenizer.num_classes}, blank index {tokenizer.blank}")

    lm_dataset_config = config.data_config.lm_dataset_config
    if not lm_dataset_config.data_paths:
        raise ValueError(
            "No LM training data. Point `data_config.lm_dataset_config.data_paths` at the ASR transcript .tsv files -- "
            "the internal LM must be fitted on the text the transducer itself trained on."
        )
    lm_dataset = datasets.get_lm(tokenizer=tokenizer, dataset_config=lm_dataset_config)

    lm: LanguageModel = BaseModel.build_lm(model_config)
    lm.summary()
    if not hasattr(lm, "fit_counts"):
        raise ValueError(
            f"{type(lm).__name__} is trained by gradient descent, not by counting, so it cannot be fitted here. "
            f"LODR wants a cheap low-order n-gram; use `BigramLanguageModel` for lm_config.internal_config."
        )

    logger.info("Counting n-grams over the LM dataset (data_config.lm_dataset_config)")
    counts = lm.fit_counts(lm_dataset.token_generator())
    # What a counting fit reports depends on how it stores the result. A bigram returns its raw
    # `[V, V]` count matrix, which is worth describing as a table; anything sparser returns a dict of
    # whatever it considers interesting, because there is no table to be a percentage of.
    if isinstance(counts, np.ndarray):
        vocab_size, seen_pairs = counts.shape[0], int((counts > 0).sum())
        logger.info(
            f"Counted {int(counts.sum())} bigrams: {seen_pairs} distinct pairs "
            f"({100.0 * seen_pairs / vocab_size**2:.2f}% of the table), "
            f"{int((counts.sum(axis=1) > 0).sum())}/{vocab_size} contexts seen"
        )
    else:
        logger.info(f"Fitted {type(lm).__name__}: " + ", ".join(f"{key}={value}" for key, value in dict(counts).items()))

    return save_lm(lm, modeldir, "internal")


if __name__ == "__main__":
    cli_util.run(main)
