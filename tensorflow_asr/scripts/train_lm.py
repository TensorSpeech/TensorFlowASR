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
import math
import os
import time

import numpy as np

from tensorflow_asr import callbacks as asr_callbacks
from tensorflow_asr import datasets, keras, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel, shift_tokens
from tensorflow_asr.utils import cli_util, env_util, file_util

logger = logging.getLogger(__name__)

TARGETS = ("external", "internal")
LR_SCHEDULES = ("cosine", "constant")


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


def to_training_pairs(
    tokens: tf.data.Dataset,
    blank: int,
    batch_size: int,
    max_length: int,
    shuffle_buffer: int = 0,
    repeat: bool = False,
    padded_length: int = None,
    drop_remainder: bool = False,
):
    """
    Turn a `tf.data` stream of token vectors into `(inputs, targets, sample_weight)` batches.

    Targets are the transcript; inputs are the same sequence shifted right with blank in front, so
    position `u` predicts token `u` from everything before it -- the same conditioning the beam
    search hands to `call_next`.

    `sample_weight` is 1 on real tokens and 0 on padding, so the loss ignores the padding. It is
    built *before* batching, as an all-ones vector the same length as the sequence, and then padded
    with 0 by `padded_batch`. Deriving it afterwards would be impossible: blank is the pad value
    *and* a legal token, since it doubles as start of sentence.

    `shuffle_buffer` matters more than it looks. Both sources arrive in a fixed order -- a text
    corpus is laid out by source document, and `transcript_tokens` turns dataset shuffling off --
    so without it the model spends thousands of consecutive steps inside one narrow slice of the
    data. Shuffling happens on single sequences before batching, so batches are mixed too.

    `repeat` cycles the data so a finite dataset can fill fixed-size epochs. It is applied **after**
    batching on purpose: repeating the sequences first would let batches straddle the seam, so a
    cycle would be `N / batch_size` batches with the remainder swallowed into the next pass.
    Repeating whole batches instead makes one cycle exactly `ceil(N / batch_size)` batches, which
    is what lets `steps_per_epoch` mean "one epoch is one pass over the data". The shuffle is
    upstream of the repeat, so every pass is shuffled differently.

    `padded_length` and `drop_remainder` exist for XLA, which compiles per input shape. Left alone,
    every batch is padded to its own longest sequence, so nearly every batch is a new shape -- fine
    on a GPU, ruinous on a TPU, where the run would spend its time recompiling. Pinning the length
    and dropping the short final batch makes every batch identically shaped, at the cost of padding
    short sequences out to `padded_length` and skipping up to `batch_size - 1` sequences per pass
    (different ones each pass, since the shuffle is upstream).
    """
    sequence_shape = [padded_length] if padded_length else [None]
    dataset = tokens.map(lambda t: t[:max_length], num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda t: tf.size(t) > 0)  # blank lines carry no supervision
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda t: (t, tf.ones_like(t, dtype=tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(sequence_shape, sequence_shape),  # [None] pads to the longest in the batch
        padding_values=(tf.constant(blank, tf.int32), 0.0),
        drop_remainder=drop_remainder,
    )
    dataset = dataset.map(lambda t, w: (shift_tokens(t, blank), t, w), num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def count_elements(dataset: tf.data.Dataset) -> int:
    """Number of elements in a dataset, by walking it once."""
    return int(dataset.reduce(tf.constant(0, tf.int64), lambda total, *_: total + 1))


def count_text_lines(text_path: str, max_lines: int = None) -> int:
    """
    Number of non-blank lines in a text corpus.

    Counted on the raw lines rather than on the tokenized stream: tokenizing tens of millions of
    lines purely to count them would cost about as much as an epoch of training. The blank-line
    filter is applied here too so the number matches what `to_training_pairs` will actually yield,
    give or take a line that survives the strip but still tokenizes to nothing.
    """
    path = file_util.preprocess_paths(text_path)
    dataset = tf.data.TextLineDataset(path, compression_type="GZIP" if str(path).endswith(".gz") else "")
    if max_lines:
        dataset = dataset.take(max_lines)
    return count_elements(dataset.filter(lambda line: tf.strings.length(tf.strings.strip(line)) > 0))


def steps_for_one_pass(num_sequences: int, batch_size: int, drop_remainder: bool = False) -> int:
    """
    Batches in one full pass, so an epoch covers the data exactly once.

    `drop_remainder` has to match the data pipeline. Rounding up when the short final batch is
    being dropped would ask for a step the pass does not contain, and the epoch would quietly
    borrow from the next one.
    """
    if drop_remainder:
        return max(num_sequences // batch_size, 1)
    return max(math.ceil(num_sequences / batch_size), 1)


class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """
    Cross entropy averaged over the real tokens rather than over every padded position.

    Keras reduces with `sum_over_batch_size`, which divides by the element count. Padded positions
    contribute nothing to the numerator but still count in the denominator, so the stock loss comes
    out scaled by the fraction of the batch that is real: at 50% padding it reports half the true
    per-token cross entropy. Two consequences, both bad. The number is not comparable to any
    published perplexity, and the gradient is scaled per batch by however much padding that batch
    happened to contain, so the effective step size wobbles with sequence length.

    Dividing by `sum(sample_weight)` fixes both. `__call__` is overridden rather than `call`
    because the division has to happen after the weights are applied, which is exactly the step
    the base class owns.

    The token count is summed **across replicas**. Keras adds up what each replica's loss returns,
    so dividing by the local count would make both the reported loss and the gradient scale with
    the number of replicas -- on a TPU v3-8 that is 8x, which silently multiplies the effective
    learning rate and makes `clipnorm` bite eight times harder. Measured before the all-reduce was
    added: two replicas reported 6.79 against a `ln(30) = 3.40` uniform baseline, exactly double.
    """

    def __init__(self, name="masked_sparse_categorical_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # `call` returns log-probabilities, and log_softmax is idempotent, so from_logits=True
        # re-normalises a distribution that is already normalised -- a no-op, not a second softmax.
        losses = keras.ops.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        weights = keras.ops.ones_like(losses) if sample_weight is None else keras.ops.cast(sample_weight, losses.dtype)

        total = keras.ops.sum(losses * weights)
        count = keras.ops.sum(weights)

        # Outside a strategy this is the default replica context and `all_reduce` is a no-op, so
        # the single-device number is unchanged.
        replica_context = tf.distribute.get_replica_context()
        if replica_context is not None:
            count = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, count)

        return total / keras.ops.maximum(count, 1.0)


def build_callbacks(modeldir: str, kaggle_model_handle: str = None, save_freq="epoch"):
    """
    Checkpointing, so a run that is cut short can be picked up rather than started again.

    This matters on a time-boxed machine. A full pass over a corpus like LibriSpeech LM is far
    longer than a Kaggle session, and `/kaggle/working` starts empty on every run -- so without
    this an interrupted run loses everything, since the weights are only written once `fit`
    returns. `KaggleModelBackupAndRestore` round-trips the checkpoint through a Kaggle Model, which
    is the same mechanism the ASR trainer uses (see `{{ kaggle_model_handle }}` in the example
    configs); the next session downloads it in `on_train_begin` and resumes.

    Without a handle there is no checkpointing at all, which is the right default for a short run
    where uploading every epoch would cost more than restarting.
    """
    if not kaggle_model_handle:
        return []
    if not modeldir:
        raise ValueError("--kaggle-model-handle needs --modeldir as well: the checkpoint is written there before being uploaded.")
    logger.info(f"Backing up to the Kaggle model {kaggle_model_handle} every {save_freq}, and restoring from it if it already exists")
    return [asr_callbacks.KaggleModelBackupAndRestore(model_dir=modeldir, model_handle=kaggle_model_handle, save_freq=save_freq)]


def build_optimizer(learning_rate: float, total_steps: int, warmup_steps: int, clipnorm: float, lr_schedule: str):
    """
    Adam, optionally with a warmup-then-cosine schedule and global-norm clipping.

    A 2x2048 LSTM at these sizes reaches gradient norms in the tens within a few dozen steps, which
    is the usual way an LSTM language model stalls: one bad step moves the weights somewhere the
    optimiser then spends thousands of steps crawling out of. Clipping the global norm is the
    standard guard, and the warmup keeps the first steps -- when Adam's second-moment estimate is
    still nearly empty and its effective step is largest -- from being the damaging ones.

    Cosine decay needs to know where the end is, so it only applies when the step budget is known
    (`steps_per_epoch` x `epochs`). Without one the learning rate is left flat rather than guessed.
    """
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule}")

    schedule = learning_rate
    if lr_schedule == "cosine":
        if total_steps:
            # Cap the warmup on short runs, otherwise a quick job is nothing but warmup.
            warmup = min(warmup_steps, max(total_steps // 10, 1))
            schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.0,
                decay_steps=max(total_steps - warmup, 1),
                warmup_target=learning_rate,
                warmup_steps=warmup,
            )
            logger.info(f"Learning rate: 0 -> {learning_rate} over {warmup} steps, then cosine to 0 at step {total_steps}")
        else:
            logger.warning(
                "lr_schedule=cosine needs a step budget to decay over. Pass --steps-per-epoch, "
                f"or --lr-schedule=constant to silence this. Holding the rate at {learning_rate}."
            )

    return keras.optimizers.Adam(learning_rate=schedule, clipnorm=clipnorm if clipnorm and clipnorm > 0 else None)


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
    steps_per_epoch: int = None,
    max_length: int = 256,
    learning_rate: float = 1e-3,
    lr_schedule: str = "cosine",
    warmup_steps: int = 1000,
    clipnorm: float = 1.0,
    shuffle_buffer: int = 10000,
    kaggle_model_handle: str = None,
    device_type: str = "gpu",
    devices: list = None,
    tpu_address: str = None,
    tpu_vm: bool = False,
    spx: int = 1,
    mxp: str = "none",
    repodir: str = os.getcwd(),
    verbose: int = 1,
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

    Note on the loss: `LanguageModel.call` returns log-probabilities, and cross entropy
    `from_logits=True` is the correct pairing for those -- `softmax(ln p) = p` whenever `p` is
    already normalised, so the loss re-normalising is a no-op rather than a second softmax. It is
    averaged over real tokens only; see `MaskedSparseCategoricalCrossentropy` for why the stock
    reduction reports the wrong number here.

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
    steps_per_epoch : int
        Steps per epoch. Left unset, the dataset is counted and this becomes
        `ceil(sequences / bs)`, so **one epoch is one full pass over the data** -- and `epochs` is
        then the number of passes.

        Set it explicitly to skip the counting pass, or to cut a corpus too large to traverse into
        shorter epochs. That is worth doing on something like the 40M-line LibriSpeech LM corpus:
        a full pass there is ~1.25M steps, and Keras reports the *running mean* of the loss over
        the current epoch, so a single enormous epoch shows a number that stops moving long before
        training does. Shorter epochs reset that average and give you a reading per epoch.

        Note that epochs do not restart the stream -- with the data cycling, epoch 2 continues
        where epoch 1 stopped. Epoch boundaries line up with passes only because the step count
        matches one pass exactly.
    lr_schedule : str
        "cosine" (default) warms up from 0 then decays to 0 over `steps_per_epoch` x `epochs`;
        "constant" holds `learning_rate`. Cosine needs `steps_per_epoch` and warns without it.
    warmup_steps : int
        Steps to reach `learning_rate`. Capped at a tenth of the run so short jobs are not all
        warmup.
    clipnorm : float
        Clip gradients to this global norm. 0 disables. Large LSTMs here reach norms in the tens
        within a few dozen steps, which is the usual cause of a language model that stalls.
    shuffle_buffer : int
        Sequences buffered for shuffling. 0 disables, which leaves the corpus in file order --
        thousands of consecutive steps inside one document. Costs roughly
        `shuffle_buffer x max_length x 4` bytes.
    kaggle_model_handle : str
        Kaggle model to check the training state in and out of, e.g.
        "owner/tensorflowasr-lm/keras/external". Needs `--modeldir`.

        Turn this on for any run longer than the machine it is on. A checkpoint goes up after
        every epoch and comes back down at the start of the next run, so an interrupted run
        continues instead of restarting -- which matters because the weights are only written to
        `--output` once `fit` returns, and a killed session otherwise loses the lot. The ASR
        trainer uses the same callback via `{{ kaggle_model_handle }}` in the example configs.

        Uploading needs write credentials, which is not the same as being able to read public
        models: set KAGGLE_USERNAME and KAGGLE_KEY, or have ~/.kaggle/kaggle.json. Inside a Kaggle
        notebook that means attaching your API token as a Secret -- the notebook's own implicit
        auth is not enough.
    bs : int
        Batch size **per replica**. The dataset is batched at `bs x replicas`, matching
        `scripts/train.py`, so a TPU v3-8 with `--bs=32` runs a global batch of 256.
    device_type : str
        "gpu" (default), "cpu" or "tpu".

        On TPU two pipeline changes are forced, because XLA compiles per input shape and the
        default pipeline produces a new shape almost every batch: sequences are padded to
        `max_length` rather than to the longest in the batch, and the short final batch is
        dropped. Both cost something -- short sequences carry padding out to `max_length`, and up
        to `bs x replicas - 1` sequences are skipped per pass -- and neither is worth paying on a
        GPU, where dynamic shapes are free.

        A stacked LSTM is a poor fit for a TPU regardless: it is sequential over timesteps, which
        is what TPUs are worst at. Measure against the GPU before committing to a session.
    tpu_address : str
        Cluster address. Leave unset on a Kaggle TPU VM.
    tpu_vm : bool
        True on a TPU VM, which skips `experimental_connect_to_cluster`. Kaggle's TPUs are VMs.
    spx : int
        `steps_per_execution`, batches per device call. Raising it cuts host round trips and is the
        usual throughput lever on TPU.

        **Left at 1 because it could not be verified.** With `keras 3` on `tensorflow 2.19`, any
        value above 1 combined with a distribution strategy fails during `fit` with
        `InvalidArgumentError: You must feed a value for placeholder tensor .../while/cond/...`.
        Measured with `MirroredStrategy` over two virtual CPU devices: it fails for a plain Dense
        model as readily as for this LSTM, and with the stock Keras loss as readily as with the
        masked one, so it is not something about this script. Single-device runs are fine at any
        value. Whether `TPUStrategy` shares the fault is untested -- there is no TPU here. Try it
        on a real TPU by all means, but check a couple of steps run before spending a session.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target}")
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule}")
    if kaggle_model_handle and not modeldir:
        # Checked here as well as in build_callbacks so it fails before the corpus is counted,
        # which on a large one is minutes of work thrown away.
        raise ValueError("--kaggle-model-handle needs --modeldir as well: the checkpoint is written there before being uploaded.")
    if text_path and target != "external":
        # The internal LM approximates what the transducer picked up from its training transcripts.
        # Counting it over any other corpus would make the correction subtract the wrong thing, so
        # there is deliberately no way to point it at one.
        raise ValueError(
            f"--text-path is only valid with --target=external, got --target={target}. The internal LM must be fitted on the ASR training transcripts."
        )

    strategy = env_util.setup_strategy(device_type=device_type, devices=devices, tpu_address=tpu_address, tpu_vm=tpu_vm)
    env_util.setup_seed()
    env_util.setup_mxp(mxp=mxp)

    # XLA compiles per input shape, and the default pipeline pads each batch to its own longest
    # sequence -- a new shape almost every step. On a GPU that is free; on a TPU it means
    # recompiling instead of training, so shapes are pinned there.
    on_tpu = device_type.lower() == "tpu"
    global_batch_size = bs * strategy.num_replicas_in_sync
    if strategy.num_replicas_in_sync > 1:
        logger.info(f"{strategy.num_replicas_in_sync} replicas: --bs={bs} per replica gives a global batch of {global_batch_size}")
    if on_tpu:
        logger.info(f"TPU: padding every sequence to max_length={max_length} and dropping the short final batch, so every step has one shape")

    config = Config(config_path, training=False, repodir=repodir, datadir=datadir, modeldir=modeldir, **kwargs)
    model_config = config.lm_config.external_config if target == "external" else config.lm_config.internal_config
    if not model_config or not model_config.get("class_name"):
        raise ValueError(f"`lm_config.{target}_config` is not set in {config_path}, nothing to train")

    tokenizer = tokenizers.get(config)
    tokenizer.make()
    logger.info(f"Vocabulary size {tokenizer.num_classes}, blank index {tokenizer.blank}")

    # Everything that creates variables goes inside the scope: under `TPUStrategy` a model built
    # outside it is not replicated across the cores, and the optimizer slots compile follows.
    with strategy.scope():
        lm: LanguageModel = BaseModel.build_lm(model_config)
        lm.summary()

        if hasattr(lm, "fit_counts"):
            # An n-gram's maximum likelihood estimate is a ratio of counts, exact and available in
            # one pass. Gradient descent would only approximate what this computes outright.
            logger.info(f"{type(lm).__name__} provides `fit_counts`, fitting by counting rather than gradient descent")
            if on_tpu:
                logger.warning("Counting runs on the host, so --device-type=tpu buys this model nothing.")
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
            if steps_per_epoch is None:
                # One epoch = one pass over the data. Counting costs a walk over the corpus, which
                # is cheap for transcripts and a decompress-and-split for a text file; pass
                # --steps-per-epoch to skip it and define the epoch yourself.
                logger.info("Counting the dataset so that one epoch is one full pass (pass --steps-per-epoch to skip)")
                started = time.time()
                num_sequences = count_text_lines(text_path, max_lines) if text_path else count_elements(tokens)
                steps_per_epoch = steps_for_one_pass(num_sequences, global_batch_size, drop_remainder=on_tpu)
                logger.info(
                    f"{num_sequences:,} sequences -> {steps_per_epoch:,} steps per epoch at global batch size "
                    f"{global_batch_size} (counted in {time.time() - started:.1f}s)"
                )

            total_steps = steps_per_epoch * epochs
            logger.info(
                f"Training {type(lm).__name__} ({lm.count_params() / 1e6:.1f}M params) on {source} "
                f"for {epochs} epochs x {steps_per_epoch:,} steps = {total_steps:,} steps"
            )
            pairs = to_training_pairs(
                tokens,
                tokenizer.blank,
                batch_size=global_batch_size,
                max_length=max_length,
                shuffle_buffer=shuffle_buffer,
                repeat=True,
                padded_length=max_length if on_tpu else None,
                drop_remainder=on_tpu,
            )
            lm.compile(
                optimizer=build_optimizer(
                    learning_rate=learning_rate,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    clipnorm=clipnorm,
                    lr_schedule=lr_schedule,
                ),
                loss=MaskedSparseCategoricalCrossentropy(),
                steps_per_execution=spx,
            )
            lm.fit(
                pairs,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=verbose,
                callbacks=build_callbacks(modeldir, kaggle_model_handle),
            )

    output = file_util.preprocess_paths(output)
    lm.save_weights(output)
    logger.info(f"Wrote {target} language model weights to {output}")


if __name__ == "__main__":
    cli_util.run(main)
