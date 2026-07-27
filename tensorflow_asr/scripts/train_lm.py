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


def token_dataset(sequences, blank: int, batch_size: int, max_length: int):
    """
    A `tf.data` pipeline of `(inputs, targets, sample_weight)` for teacher-forced LM training.

    Targets are the transcript; inputs are the same sequence shifted right with blank in front, so
    position `u` predicts token `u` from everything before it -- the same conditioning the beam
    search hands to `call_next`.

    `sample_weight` is 0 on padding so the loss ignores it. It cannot be derived from the values:
    blank is the pad value *and* a legal token, since it doubles as start of sentence.
    """

    def generator():
        for seq in sequences:
            seq = np.asarray(seq, dtype=np.int32).reshape(-1)[:max_length]
            targets = np.full([max_length], blank, dtype=np.int32)
            targets[: seq.size] = seq
            weights = np.zeros([max_length], dtype=np.float32)
            weights[: seq.size] = 1.0
            yield targets, weights

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec([max_length], tf.int32), tf.TensorSpec([max_length], tf.float32)),
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda targets, weights: (shift_tokens(targets, blank), targets, weights))
    return dataset.prefetch(tf.data.AUTOTUNE)


def main(
    config_path: str,
    datadir: str,
    dataset_type: str,
    output: str,
    target: str = "internal",
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
    - "external" builds `lm_config.external_config`, the LM that gets fused in. Training it on the
      ASR transcripts only reproduces what the model already knows -- the point of an external LM
      is a far larger corpus -- so point `data_config.train_dataset_config` at that text.

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
        Where to write the h5. Point `lm_config.<target>_config.weights` at it.
    max_length : int
        Transcripts are truncated to this many tokens. Only affects gradient training.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target}")

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

    sequences = transcript_tokens(tokenizer, dataset_type, config.data_config.train_dataset_config)

    if hasattr(lm, "fit_counts"):
        # An n-gram's maximum likelihood estimate is a ratio of counts, exact and available in one
        # pass. Gradient descent would only approximate what this computes outright.
        logger.info(f"{type(lm).__name__} provides `fit_counts`, fitting by counting rather than gradient descent")
        counts = lm.fit_counts(sequences)
        vocab_size, seen_pairs = counts.shape[0], int((counts > 0).sum())
        logger.info(
            f"Counted {int(counts.sum())} bigrams: {seen_pairs} distinct pairs "
            f"({100.0 * seen_pairs / vocab_size**2:.2f}% of the table), "
            f"{int((counts.sum(axis=1) > 0).sum())}/{vocab_size} contexts seen"
        )
    else:
        sequences = list(sequences)
        logger.info(f"Training {type(lm).__name__} on {len(sequences)} transcripts for {epochs} epochs")
        lm.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        lm.fit(token_dataset(sequences, tokenizer.blank, batch_size=bs, max_length=max_length), epochs=epochs)

    output = file_util.preprocess_paths(output)
    lm.save_weights(output)
    logger.info(f"Wrote {target} language model weights to {output}")


if __name__ == "__main__":
    cli_util.run(main)
