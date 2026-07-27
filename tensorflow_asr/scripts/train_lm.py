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

import numpy as np

from tensorflow_asr import datasets, keras, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel, shift_tokens
from tensorflow_asr.utils import cli_util, env_util, file_util

logger = logging.getLogger(__name__)

TARGETS = ("external", "internal")


def transcript_tokens(tokenizer, dataset_type: str, dataset_config):
    """
    Tokenized transcripts of a dataset, one array per utterance.

    Reads the entries directly rather than building the audio pipeline: a language model needs the
    text and nothing else, and loading the waveforms would dominate the runtime for no reason.
    """
    from tqdm import tqdm

    dataset_config.shuffle = False
    dataset_config.drop_remainder = False
    dataset = datasets.get(tokenizer=tokenizer, dataset_config=dataset_config, dataset_type=dataset_type)
    dataset.read_entries()
    for text in tqdm(dataset.vocab_generator(), total=dataset.num_entries, desc="Reading transcripts"):
        yield tokenizer.tokenize(text).numpy()


def to_training_pairs(tokens: tf.data.Dataset, blank: int, batch_size: int, max_length: int):
    """
    Turn a `tf.data` stream of token vectors into `(inputs, targets, sample_weight)` batches.

    Targets are the transcript; inputs are the same sequence shifted right with blank in front, so
    position `u` predicts token `u` from everything before it -- the same conditioning the beam
    search hands to `call_next`.

    `sample_weight` is 1 on real tokens and 0 on padding, so the loss ignores the padding. It is
    built *before* batching, as an all-ones vector the same length as the sequence, and then padded
    with 0 by `padded_batch`. Deriving it afterwards would be impossible: blank is the pad value
    *and* a legal token, since it doubles as start of sentence.
    """
    dataset = tokens.map(lambda t: t[:max_length], num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda t: tf.size(t) > 0)  # blank lines carry no supervision
    dataset = dataset.map(lambda t: (t, tf.ones_like(t, dtype=tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None]),  # to the longest in the batch, not to `max_length`
        padding_values=(tf.constant(blank, tf.int32), 0.0),
    )
    dataset = dataset.map(lambda t, w: (shift_tokens(t, blank), t, w), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def text_line_tokens(tokenizer, text_path: str, max_lines: int = None) -> tf.data.Dataset:
    """
    Tokenize a plain text corpus, one sentence per line, as a streaming `tf.data` pipeline.

    This is the path for the **external** language model, whose whole point is a corpus far larger
    than the ASR transcripts -- the LibriSpeech LM corpus is ~40M lines and 800M words, several GB
    uncompressed. So nothing is materialised: `TextLineDataset` streams the file (transparently
    through gzip when the name ends in `.gz`, which is how OpenSLR ships it) and tokenization runs
    inside the graph in parallel. Reading it into a python list first, as the transcript path can
    afford to, would need tens of GB and hours of eager op dispatch.

    Tokenizing with the tokenizer built from your own config is what guarantees the indices match
    the transducer's vocabulary -- the requirement `LanguageModel.call_next` states but cannot
    check.
    """
    path = file_util.preprocess_paths(text_path)
    dataset = tf.data.TextLineDataset(
        path,
        compression_type="GZIP" if str(path).endswith(".gz") else "",
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    if max_lines:
        dataset = dataset.take(max_lines)
    return dataset.map(lambda line: tf.cast(tokenizer.tokenize(line), tf.int32), num_parallel_calls=tf.data.AUTOTUNE)


def transcript_token_dataset(sequences) -> tf.data.Dataset:
    """The transcript generator as a `tf.data` stream, so it can share `to_training_pairs`."""

    def generator():
        for seq in sequences:
            yield np.asarray(seq, dtype=np.int32).reshape(-1)

    return tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec([None], tf.int32))


def main(
    config_path: str,
    datadir: str,
    dataset_type: str,
    output: str,
    target: str = "internal",
    text_path: str = None,
    max_lines: int = None,
    modeldir: str = None,
    bs: int = 32,
    epochs: int = 10,
    max_length: int = 256,
    learning_rate: float = 1e-3,
    device_type: str = "gpu",
    devices: list = None,
    mxp: str = "none",
    repodir: str = os.getcwd(),
    **kwargs,
):
    """
    Train a beam search language model on transcripts.

    Which model is trained is chosen by `target`, naming the key of `lm_config` it builds:

    - "internal" (default) builds `lm_config.internal_config`, the low-order LM that LODR
      subtracts. This one **must** be fitted on the ASR training transcripts, because what it
      approximates is the internal LM the transducer picked up from exactly that text. Fitting it
      on target-domain text would make the correction subtract the knowledge fusion is adding.
    - "external" builds `lm_config.external_config`, the LM that gets fused in. Point
      `--text-path` at a large text corpus: the whole value of an external LM is seeing far more
      text than the ASR transcripts. The published setups use the LibriSpeech LM corpus, ~40M
      lines and 800M words, against the ~9M words of LibriSpeech transcripts:

          wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz

      `.gz` is read directly, no need to decompress. Without `--text-path` it falls back to the
      transcripts and warns, because that trains the external LM on exactly the text the
      transducer already learned -- which is what ILME and LODR exist to *subtract*.

    Fitting dispatches on the model. An n-gram exposes `fit_counts` and is fitted in a single
    counting pass, which is its exact maximum-likelihood estimate; anything else is trained by
    gradient descent on next-token cross-entropy. Either way the result is written to `output` as
    h5, ready for the `weights` key of the config that built it.

    Note on the loss: `LanguageModel.call` returns log-probabilities, and
    `SparseCategoricalCrossentropy(from_logits=True)` is the correct pairing for those --
    `softmax(ln p) = p` whenever `p` is already normalised, so the loss re-normalising is a no-op
    rather than a second softmax.

    Parameters
    ----------
    output : str
        Where to write the h5. Pass it to `tensorflow_asr test` as `--lm-h5` or `--internal-lm-h5`.
    text_path : str
        Plain text corpus, one sentence per line, optionally gzipped. External LM only.
    max_lines : int
        Stop after this many lines of `--text-path`. For a quick run over a corpus of tens of
        millions of lines.
    max_length : int
        Sequences are truncated to this many tokens. Only affects gradient training.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target}")
    if text_path and target != "external":
        # The internal LM approximates what the transducer picked up from its training transcripts.
        # Counting it over any other corpus would make the correction subtract the wrong thing, so
        # there is deliberately no way to point it at one.
        raise ValueError(f"--text-path is only valid with --target=external, got --target={target}. The internal LM must be fitted on the ASR training transcripts.")

    env_util.setup_strategy(device_type=device_type, devices=devices)
    env_util.setup_seed()
    env_util.setup_mxp(mxp=mxp)

    config = Config(config_path, training=False, repodir=repodir, datadir=datadir, modeldir=modeldir, **kwargs)
    model_config = config.lm_config.external_config if target == "external" else config.lm_config.internal_config
    if not model_config or not model_config.get("class_name"):
        raise ValueError(f"`lm_config.{target}_config` is not set in {config_path}, nothing to train")

    tokenizer = tokenizers.get(config)
    tokenizer.make()
    logger.info(f"Vocabulary size {tokenizer.num_classes}, blank index {tokenizer.blank}")

    lm: LanguageModel = BaseModel.build_lm(model_config)
    lm.summary()

    if hasattr(lm, "fit_counts"):
        # An n-gram's maximum likelihood estimate is a ratio of counts, exact and available in one
        # pass. Gradient descent would only approximate what this computes outright.
        logger.info(f"{type(lm).__name__} provides `fit_counts`, fitting by counting rather than gradient descent")
        counts = lm.fit_counts(transcript_tokens(tokenizer, dataset_type, config.data_config.train_dataset_config))
        vocab_size, seen_pairs = counts.shape[0], int((counts > 0).sum())
        logger.info(
            f"Counted {int(counts.sum())} bigrams: {seen_pairs} distinct pairs "
            f"({100.0 * seen_pairs / vocab_size**2:.2f}% of the table), "
            f"{int((counts.sum(axis=1) > 0).sum())}/{vocab_size} contexts seen"
        )
    else:
        if text_path:
            tokens = text_line_tokens(tokenizer, text_path, max_lines=max_lines)
            source = f"{text_path}{f' (first {max_lines} lines)' if max_lines else ''}"
        else:
            if target == "external":
                logger.warning(
                    "Training the external language model on the ASR transcripts, which is the text the transducer "
                    "already learned. Pass --text-path to a larger corpus, or the fusion has little left to add."
                )
            tokens = transcript_token_dataset(transcript_tokens(tokenizer, dataset_type, config.data_config.train_dataset_config))
            source = "the training transcripts"
        logger.info(f"Training {type(lm).__name__} ({lm.count_params() / 1e6:.1f}M params) on {source} for {epochs} epochs")
        lm.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        lm.fit(to_training_pairs(tokens, tokenizer.blank, batch_size=bs, max_length=max_length), epochs=epochs)

    output = file_util.preprocess_paths(output)
    lm.save_weights(output)
    logger.info(f"Wrote {target} language model weights to {output}")


if __name__ == "__main__":
    cli_util.run(main)
