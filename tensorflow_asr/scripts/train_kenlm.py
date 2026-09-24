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
"""Build the external n-gram language model with KenLM, and convert it into GPU tensors"""

import logging
import os

from tensorflow_asr import datasets, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel, lm_dir, save_lm
from tensorflow_asr.utils import cli_util, file_util

logger = logging.getLogger(__name__)


def main(
    config_path: str,
    datadir: str,
    modeldir: str,
    prune: list = None,
    arpa: str = None,
    text_path: str = None,
    max_lines: int = None,
    lmplz: str = None,
    lmplz_args: list = None,
    overwrite_text: bool = False,
    kaggle_model_handle: str = None,
    repodir: str = os.getcwd(),
    **kwargs,
):
    """
    Build a token-level n-gram with KenLM and load it into `lm_config.external_config`.

    This is the path NGPU-LM (https://arxiv.org/abs/2505.22857) itself takes, and the only one that
    scales. `train_internal_lm` counts n-grams in python dicts, roughly 300-400 bytes per token, so
    it tops out around a few million tokens -- fine for ASR transcripts, hopeless for a corpus like
    the 800M-word LibriSpeech LM set. `lmplz` does the same job by disk-based merge sort in bounded
    memory.

    Three things happen, and `<modeldir>/lm` keeps all of them::

        lm/corpus.ids.txt      the corpus tokenized to token ids, what lmplz reads
        lm/lm.arpa             what lmplz writes
        lm/kenlm.weights.h5    the arc tensors, what decoding loads

    The intermediate text is kept rather than piped, because tokenising a large corpus is the slow
    half; a second run at a different pruning reuses it. Pass `--overwrite-text` to rebuild it --
    which you must whenever the corpus or the tokenizer has changed, since nothing detects that.

    **The order is not a flag.** It comes from `order` in `lm_config.external_config`, because that
    is the value that survives into the h5 and drives how many times the backoff walk unrolls at
    decoding time. Passing it here as well would only be a second place for the two to disagree.

    Like `train_external_lm` this writes `lm_config.external_config`; the two are alternatives, an
    n-gram or a neural LM. `--arpa` skips the build and converts an ARPA you already have, which
    must still be **token-level over this config's tokenizer** -- a word-level ARPA is rejected.

    Parameters
    ----------
    config_path : str
        The same config the ASR model uses, so the tokenizer -- and therefore the ids in the corpus
        and the vocabulary the LM is indexed against -- matches. Its
        `lm_config.external_config.config.order` is the n-gram order.
    datadir : str
    modeldir : str
        Everything is written under `<modeldir>/lm`: the tokenized corpus, the ARPA, and
        `kenlm.weights.h5`. Pass that last one to `tensorflow_asr test --lm-h5`.
    prune : Optional[list]
        `lmplz --prune`, one count cutoff per order, eg. `[0,0,1,1]`. This is where to shrink the
        model: cheaper and better informed than having the reader drop arcs to fit `max_arcs`.
    arpa : Optional[str]
        Convert this ARPA instead of building one. Skips the corpus entirely.
    text_path : Optional[str]
        Where the tokenized token-id corpus that `lmplz` reads is written, and reused from on a
        later run (see `--overwrite-text`). Defaults to `<modeldir>/lm/corpus.ids.txt`. Point it at
        a roomier volume when the corpus is large -- the ids file is about the size of the source
        text, and on a small `/kaggle/working` it can be worth keeping off it.
    max_lines : Optional[int]
        Stop after this many lines, for a trial run over a huge corpus.
    lmplz : Optional[str]
        The KenLM binary. Left unset it is looked for on PATH and then at
        `externals/kenlm/build/bin/lmplz`, where `./scripts/install_kenlm.sh` puts it -- so a
        standard install needs nothing here, whether or not that directory is on PATH. An explicit
        path is used as given and never falls back, so a typo fails loudly.
    lmplz_args : Optional[list]
        Extra flags, eg. `["-S", "40%"]` to cap memory, or `["--discount_fallback"]` which small
        corpora need when an order has too few n-grams to estimate a discount from.
    overwrite_text : bool
        Re-tokenise the corpus even when `lm/corpus.ids.txt` is already there. Reuse is the default
        because tokenising is the slow half, and between runs that only change `--prune` it is the
        right call. Pass this whenever the ids would come out different -- the corpus changed,
        `data_paths` changed, `--max-lines` changed, or the tokenizer did. A stale file is still a
        valid file, so nothing catches it, and the model would end up indexed against a vocabulary
        the transducer no longer emits.
    kaggle_model_handle : Optional[str]
        Upload the finished model to this Kaggle model as a new version, eg.
        "owner/tensorflowasr-lm/keras/kenlm". This is the counting counterpart to what
        `train_external_lm` does through its checkpoint callback: there is no `fit` loop here to
        back up per epoch, so the single built model is pushed once at the end. Auto-creates the
        handle on first upload; the token-id corpus is left out as an input, not a result.
        Credentials come from `KAGGLE_USERNAME` / `KAGGLE_KEY`, which the Kaggle notebook exports.
    repodir : str
    """
    config = Config(config_path, training=False, repodir=repodir, datadir=datadir, modeldir=modeldir, **kwargs)
    model_config = config.lm_config.external_config
    if not model_config or not model_config.get("class_name"):
        raise ValueError(f"`lm_config.external_config` is not set in {config_path}, nothing to build")

    tokenizer = tokenizers.get(config)
    tokenizer.make()
    logger.info(f"Vocabulary size {tokenizer.num_classes}, blank index {tokenizer.blank}")

    lm: LanguageModel = BaseModel.build_lm(model_config)
    lm.summary()
    if not hasattr(lm, "load_arpa"):
        raise ValueError(
            f"{type(lm).__name__} cannot read an ARPA. Set lm_config.external_config to NGramLanguageModel, "
            f"or train a neural LM with `tensorflow_asr train_external_lm`."
        )
    # The ARPA's order has to match the model's, since the config value is what survives into the h5
    # reload and drives how many times the backoff walk unrolls.
    order = getattr(lm, "order", 6)

    directory = lm_dir(modeldir)
    if arpa:
        arpa_path = file_util.preprocess_paths(arpa)
        logger.info(f"Converting the existing ARPA at {arpa_path}; the corpus is not read")
    else:
        lm_dataset_config = config.data_config.lm_dataset_config.external_dataset_config
        if not lm_dataset_config.data_paths:
            raise ValueError(
                "No LM training data. Point `data_config.lm_dataset_config.external_dataset_config.data_paths` at a "
                ".txt/.txt.gz corpus, or pass --arpa=<path> to convert an n-gram you already built."
            )
        lm_dataset = datasets.get_lm(tokenizer=tokenizer, dataset_config=lm_dataset_config)
        arpa_path = lm_dataset.create_arpa(
            arpa_path=os.path.join(directory, "lm.arpa"),
            text_path=text_path or os.path.join(directory, "corpus.ids.txt"),
            order=order,
            prune=prune,
            max_lines=max_lines,
            lmplz=lmplz,
            lmplz_args=lmplz_args,
            overwrite_text=overwrite_text,
        )

    stats = lm.load_arpa(arpa_path)
    logger.info(f"Loaded {type(lm).__name__}: " + ", ".join(f"{key}={value}" for key, value in dict(stats).items()))
    weights_path = save_lm(lm, modeldir, "kenlm")

    if kaggle_model_handle:
        # Upload the modeldir, not just the h5, so the arc weights and the ARPA travel together and a
        # restore lands them back where decoding expects. The token-id corpus is excluded: it can be
        # gigabytes, and it is the input to this build, not its result.
        from tensorflow_asr.callbacks import upload_kaggle_model  # pylint: disable=import-outside-toplevel

        upload_kaggle_model(
            modeldir,
            kaggle_model_handle,
            notes=f"kenlm n-gram: {', '.join(f'{k}={v}' for k, v in dict(stats).items())}",
            ignore_patterns=["*.ids.txt", "*.ids.txt.gz"],
        )

    return weights_path


if __name__ == "__main__":
    cli_util.run(main)
