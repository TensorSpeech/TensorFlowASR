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

os.environ["TQDM_DISABLE"] = "1"

from tensorflow_asr import callbacks as asr_callbacks
from tensorflow_asr import datasets, keras, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel
from tensorflow_asr.utils import cli_util, env_util, file_util

logger = logging.getLogger(__name__)

TARGETS = ("external", "internal")
LR_SCHEDULES = ("cosine", "constant")


def check_steps_per_epoch(steps_per_epoch: int) -> int:
    """
    Gradient training needs to be told the epoch length; it is not derived.

    Deriving it means walking the whole corpus before the first step -- minutes on a large text
    file, and the walk has to be repeatable or it consumes the data it just measured. Asking for
    the number keeps startup constant and makes the epoch an explicit choice, which it should be:
    a full pass over a 40M-line corpus is a worse epoch than a short one, because the progress bar
    reports a running mean that flattens out over a long epoch.
    """
    if not steps_per_epoch or steps_per_epoch < 1:
        raise ValueError(
            "--steps-per-epoch is required and must be at least 1. One pass over the data is "
            "ceil(sequences / (bs x replicas)) steps; `wc -l` on the corpus gives the sequence count."
        )
    return steps_per_epoch


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
        # Written out rather than calling `sparse_categorical_crossentropy`, which is the same
        # arithmetic but reaches `tf.nn.sparse_softmax_cross_entropy_with_logits`. That op adds a
        # runtime shape check whenever the static shapes are not fully known -- which is every GPU
        # batch, since those pad to the longest sequence in the batch rather than to `max_length`.
        # The check is an `Assert`, and `Assert` has no GPU kernel on any backend, so under
        # `MirroredStrategy` (which pins every op to the device) it fails outright with
        # "Could not satisfy explicit device specification" unless TF_SOFT_PLACEMENT is on.
        #
        # log_softmax then gather is the same value with no assertion in the graph. `call` already
        # returns log-probabilities and log_softmax is idempotent, so this re-normalises a
        # distribution that is already normalised -- a no-op, not a second softmax.
        #
        # `tf.gather(batch_dims=2)` rather than `keras.ops.take_along_axis`, which needs an
        # explicit `expand_dims` and lowers to `BroadcastTo` + gather. Measured pinned to the
        # device with soft placement off, the broadcast is itself unplaceable on a backend with
        # thin kernel coverage, so it trades one such op for another. `one_hot` then sum also
        # works but materialises a [batch, time, vocab] tensor to read one value per position.
        #
        # float32 regardless of the mixed-precision policy: log_softmax over the vocabulary is
        # where a float16 loss loses its accuracy, and the cast costs nothing next to the matmul.
        log_probs = keras.ops.log_softmax(keras.ops.cast(y_pred, "float32"), axis=-1)
        losses = -tf.gather(log_probs, keras.ops.cast(y_true, "int32"), batch_dims=2, axis=2)
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


def main(
    config_path: str,
    datadir: str,
    output: str,
    target: str = "internal",
    modeldir: str = None,
    bs: int = 32,
    epochs: int = 10,
    steps_per_epoch: int = None,
    learning_rate: float = 1e-3,
    lr_schedule: str = "cosine",
    warmup_steps: int = 1000,
    clipnorm: float = 1.0,
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
      on target-domain text would make the correction subtract the knowledge fusion is adding, so
      point `data_config.lm_dataset_config.data_paths` at the transcript `.tsv` files.
    - "external" builds `lm_config.external_config`, the LM that gets fused in. The whole value of
      an external LM is seeing far more text than the ASR transcripts, so point
      `data_config.lm_dataset_config.data_paths` at a large corpus. The published setups use the
      LibriSpeech LM corpus, ~40M lines and 800M words, against the ~9M words of LibriSpeech
      transcripts:

          wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz

      `.gz` is read directly, no need to decompress.

    The text, how much of it to read, and how long each sequence may be all come from
    `data_config.lm_dataset_config`: its `data_paths` (transcript `.tsv`, a `.txt`/`.txt.gz` corpus,
    or a mix -- `datasets.LMDataset` reads them all), `max_length` (tokens per sequence; sequences
    are truncated to it and it is what TPU pads to) and `max_lines` (lines to read, for a quick run
    over a huge corpus). With no `data_paths` there is nothing to train on and the run stops.

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
    steps_per_epoch : int
        Steps per epoch. **Required** for gradient training; the n-gram models, which fit by
        counting, ignore it.

        For one epoch to be one full pass over the data, set it to
        `ceil(sequences / (bs x replicas))` -- `wc -l` on the corpus gives the sequence count, and
        it is worth writing down rather than recomputing, since counting means reading the whole
        corpus before training can start.

        A full pass is often the wrong epoch anyway. The 40M-line LibriSpeech LM corpus is ~1.25M
        steps at batch 32, and Keras reports the *running mean* of the loss over the current epoch,
        so one enormous epoch shows a number that stops moving long before training does. Shorter
        epochs reset that average and give you a reading you can act on.

        Note that epochs do not restart the stream -- with the data cycling, epoch 2 continues
        where epoch 1 stopped. Epoch boundaries line up with passes only when the step count
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
        auth is not enough. The upload is not guarded: without the credentials it raises at the
        end of the first epoch and takes the run down with it, so check them before a long run.
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

    config = Config(config_path, training=False, repodir=repodir, datadir=datadir, modeldir=modeldir, **kwargs)
    model_config = config.lm_config.external_config if target == "external" else config.lm_config.internal_config
    if not model_config or not model_config.get("class_name"):
        raise ValueError(f"`lm_config.{target}_config` is not set in {config_path}, nothing to train")

    tokenizer = tokenizers.get(config)
    tokenizer.make()
    logger.info(f"Vocabulary size {tokenizer.num_classes}, blank index {tokenizer.blank}")

    # The text to train on, and every dataset knob, come from `data_config.lm_dataset_config` -- no
    # CLI override. Its `data_paths` may be ASR transcript `.tsv` (for the internal LM), a
    # `.txt`/`.txt.gz` corpus (for the external LM), or a mix; `LMDataset` reads them all and streams
    # even a multi-GB corpus rather than loading it, tokenising with this same tokenizer so the
    # indices match the transducer's vocabulary. `max_length`, `max_lines`, `shuffle`, `buffer_size`,
    # `drop_remainder` and `indefinite` ride along on the same config; `LMDataset.create` batches and
    # teacher-forces from them below.
    lm_dataset_config = config.data_config.lm_dataset_config
    if not lm_dataset_config.data_paths:
        raise ValueError(
            "No LM training data. Set `data_config.lm_dataset_config.data_paths` in the config: "
            "transcript .tsv for --target=internal, a .txt/.txt.gz corpus for --target=external."
        )
    lm_dataset = datasets.get_lm(tokenizer=tokenizer, dataset_config=lm_dataset_config)
    source = "the LM dataset (data_config.lm_dataset_config)"

    if on_tpu:
        if not lm_dataset.max_length:
            lm_dataset.compute_metadata()
        lm_dataset.drop_remainder = True

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
            logger.info(f"Counting bigrams over {source}")
            counts = lm.fit_counts(lm_dataset.token_generator())
            vocab_size, seen_pairs = counts.shape[0], int((counts > 0).sum())
            logger.info(
                f"Counted {int(counts.sum())} bigrams: {seen_pairs} distinct pairs "
                f"({100.0 * seen_pairs / vocab_size**2:.2f}% of the table), "
                f"{int((counts.sum(axis=1) > 0).sum())}/{vocab_size} contexts seen"
            )
        else:
            steps_per_epoch = check_steps_per_epoch(steps_per_epoch)
            total_steps = steps_per_epoch * epochs
            logger.info(
                f"Training {type(lm).__name__} ({lm.count_params() / 1e6:.1f}M params) on {source} "
                f"for {epochs} epochs x {steps_per_epoch:,} steps = {total_steps:,} steps"
            )
            pairs = lm_dataset.create(batch_size=global_batch_size)
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
