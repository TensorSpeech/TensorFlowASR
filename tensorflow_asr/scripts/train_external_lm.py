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
"""Train the external language model -- the neural one that gets shallow fused into the beam"""

import logging
import os

os.environ["TQDM_DISABLE"] = "1"

from tensorflow_asr import callbacks as asr_callbacks
from tensorflow_asr import datasets, keras, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel, lm_weights_path
from tensorflow_asr.utils import cli_util, env_util

logger = logging.getLogger(__name__)

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

        # `where`, not `losses * weights`. Multiplying is the obvious way to apply a mask and it is
        # wrong here: `inf * 0` and `nan * 0` are both `nan`, so a single non-finite value at a
        # position the mask exists to *ignore* still poisons the sum -- and from there the gradients,
        # and from there every weight, permanently. Padded positions are real forward passes over
        # pad tokens, so they are exactly where a value nobody is supervising can blow up unnoticed.
        # `where` never evaluates the masked branch's value into the sum, so padding cannot
        # contribute at all. This does not hide a genuinely diverged model: a non-finite value at a
        # *real* token still propagates, which is what TerminateOnNaN is here to catch.
        losses = keras.ops.where(weights > 0, losses, keras.ops.zeros_like(losses))

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
    Stop on NaN, and checkpoint so a run that is cut short can be picked up rather than started again.

    `TerminateOnNaN` is not optional here. Once the loss is NaN the weights are already NaN: nothing
    in this path can undo it, since a float32 policy attaches no `LossScaleOptimizer` to skip the
    step and `clipnorm` cannot rescue a NaN gradient (the clipped value is NaN too). Every later
    forward pass is then NaN, which survives the epoch boundary -- so without this the run keeps
    burning hours reporting `loss: nan`. Note the progress bar's own number is a running mean and
    cannot recover *within* an epoch either way; this stops the run on the first bad step instead.

    `ModelCheckpoint` writes the weights every epoch, to the same
    `<modeldir>/lm/external.weights.h5` that decoding loads. It is what makes this the *only* place
    the trained model is written -- there is no save after `fit` returns, so a run stopped partway
    through still leaves a usable language model from its last completed epoch rather than nothing
    at all. Training a language model on a real corpus is long enough that this is the normal case,
    not the exceptional one.

    `KaggleModelBackupAndRestore` is a different thing and both are wanted: it round-trips
    optimizer state through a Kaggle Model so an interrupted run can *resume mid-training*, which
    the checkpoint above cannot do. It is the same mechanism the ASR trainer uses (see
    `{{ kaggle_model_handle }}` in the example configs), and without a handle it is simply skipped.
    """
    callbacks = [
        asr_callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(filepath=lm_weights_path(modeldir, "external"), save_weights_only=True, save_freq="epoch"),
    ]
    if not kaggle_model_handle:
        return callbacks
    if not modeldir:
        raise ValueError("--kaggle-model-handle needs --modeldir as well: the checkpoint is written there before being uploaded.")
    logger.info(f"Backing up to the Kaggle model {kaggle_model_handle} every {save_freq}, and restoring from it if it already exists")
    callbacks.append(asr_callbacks.KaggleModelBackupAndRestore(model_dir=modeldir, model_handle=kaggle_model_handle, save_freq=save_freq))
    return callbacks


class NanSafeAdam(keras.optimizers.Adam):
    """
    Adam that drops non-finite gradients instead of writing them into the weights.

    Without this a single non-finite gradient is fatal and permanent. Nothing else in this path
    catches it: a float32 policy attaches no `LossScaleOptimizer` to skip the step (Keras only does
    that under `mixed_float16`), and `clipnorm` makes it worse rather than better -- the global norm
    of a vector containing NaN is NaN, so the "clipped" gradient is NaN too. Once Adam has written
    NaN into a weight every later forward pass is NaN, which survives the epoch boundary and, worse,
    gets checkpointed and restored by `KaggleModelBackupAndRestore` on the next session.

    Zeroing rather than skipping the whole step: it is one tensor op with no control flow, so it
    behaves the same eagerly, in a graph, under XLA and across replicas. Adam still decays its
    moments for that variable, which is a far smaller error than a NaN weight. This runs *before*
    `super().apply`, so the clipping inside it sees finite values and computes a real norm.

    This is a guard against transient spikes, not a way to train through a diverging model: the loss
    is untouched, so a genuine divergence still shows up as a NaN loss and `TerminateOnNaN` still
    stops the run.
    """

    def apply(self, grads, trainable_variables=None):
        grads = [None if g is None else tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        return super().apply(grads, trainable_variables)


def build_optimizer(learning_rate: float, total_steps: int, warmup_steps: int, clipnorm: float, lr_schedule: str):
    """
    Adam, optionally with a warmup-then-cosine schedule and global-norm clipping.

    A 2x2048 LSTM at these sizes reaches gradient norms in the tens within a few dozen steps, which
    is the usual way an LSTM language model stalls: one bad step moves the weights somewhere the
    optimiser then spends thousands of steps crawling out of. Clipping the global norm is the
    standard guard, and the warmup keeps the first steps -- when Adam's second-moment estimate is
    still nearly empty and its effective step is largest -- from being the damaging ones.

    `global_clipnorm`, not `clipnorm`. They sound interchangeable and are not: keras' `clipnorm`
    rescales each weight tensor on its own, so on the spikes clipping exists for it shrinks whichever
    tensors blew up and leaves the rest, which rotates the update away from the gradient. Measured on
    this model, a 40x spike came out at cosine similarity 0.59 to the true gradient under `clipnorm`
    and 1.0000 under `global_clipnorm`. One scalar over the whole gradient is what Pascanu et al.
    (2013) specify for recurrent nets, and preserving the direction is the entire point.

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

    return NanSafeAdam(learning_rate=schedule, global_clipnorm=clipnorm if clipnorm and clipnorm > 0 else None)


def main(
    config_path: str,
    datadir: str,
    modeldir: str,
    bs: int = 32,
    epochs: int = 10,
    steps_per_epoch: int = None,
    learning_rate: float = 1e-3,
    lr_schedule: str = "constant",
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
    Train `lm_config.external_config` by gradient descent, and write it to `<modeldir>/lm`.

    The external LM is the one **fused in**, weighted by `decoder_config.lm_alpha`. Its whole value
    is having seen far more text than the ASR transcripts, so point
    `data_config.lm_dataset_config.external_dataset_config.data_paths` at a large corpus. The
    published setups use the LibriSpeech LM corpus, ~40M lines and 800M words, against the ~9M words
    of transcripts::

        wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz

    `.gz` is read directly, no need to decompress.

    This script is for a **neural** external LM -- `LSTMLanguageModel` and anything else trained by
    gradient descent on next-token cross-entropy. An n-gram is not trained that way: its estimate is
    a ratio of counts, so it has its own script, `train_kenlm_lm`, which also targets
    `external_config`. The two are alternatives; pick one per config.

    Everything about the text comes from `data_config.lm_dataset_config.external_dataset_config`:
    `data_paths`, `max_length` (tokens per sequence, and what TPU pads to) and `max_lines` (for a
    quick run over a huge corpus). There is no CLI override.

    Parameters
    ----------
    config_path : str
        The same config the ASR model uses, so the tokenizer -- and therefore the vocabulary the LM
        is indexed against -- matches.
    datadir : str
    modeldir : str
        Weights are written to `<modeldir>/lm/external.weights.h5`. Pass that to
        `tensorflow_asr test --lm-h5`.
    bs : int
        Per-replica batch size.
    epochs : int
    steps_per_epoch : int
        **Required.** One pass is `ceil(sequences / (bs x replicas))` steps; `wc -l` on the corpus
        gives the sequence count. It is not derived, because deriving it means walking the whole
        corpus before the first step.
    learning_rate : float
    lr_schedule : str
        "constant" or "cosine". Cosine warms up then decays, and needs a step budget to decay over.
    warmup_steps : int
    clipnorm : float
        Global-norm clipping. An LSTM language model at these sizes reaches gradient norms in the
        tens within a few dozen steps, and one bad step is what usually stalls the run.
    kaggle_model_handle : str
        Round-trip the checkpoint through a Kaggle Model so a time-boxed session can resume, eg.
        "owner/tensorflowasr-lm/keras/external".
    verbose : int
    """
    env_util.setup_seed()
    strategy = env_util.setup_strategy(device_type=device_type, devices=devices, tpu_address=tpu_address, tpu_vm=tpu_vm)
    on_tpu = device_type.lower() == "tpu"
    env_util.setup_mxp(mxp=mxp)

    global_batch_size = bs * strategy.num_replicas_in_sync
    if strategy.num_replicas_in_sync > 1:
        logger.info(f"{strategy.num_replicas_in_sync} replicas: --bs={bs} per replica gives a global batch of {global_batch_size}")

    config = Config(config_path, training=False, repodir=repodir, datadir=datadir, modeldir=modeldir, **kwargs)
    model_config = config.lm_config.external_config
    if not model_config or not model_config.get("class_name"):
        raise ValueError(f"`lm_config.external_config` is not set in {config_path}, nothing to train")

    tokenizer = tokenizers.get(config)
    tokenizer.make()
    logger.info(f"Vocabulary size {tokenizer.num_classes}, blank index {tokenizer.blank}")

    lm_dataset_config = config.data_config.lm_dataset_config.external_dataset_config
    if not lm_dataset_config.data_paths:
        raise ValueError(
            "No LM training data. Point `data_config.lm_dataset_config.external_dataset_config.data_paths` at a "
            ".txt/.txt.gz corpus -- the whole point of an external LM is seeing more text than the ASR transcripts."
        )
    lm_dataset = datasets.get_lm(tokenizer=tokenizer, dataset_config=lm_dataset_config)

    if on_tpu:
        if not lm_dataset.max_length:
            lm_dataset.compute_metadata()
        lm_dataset.drop_remainder = True

    steps_per_epoch = check_steps_per_epoch(steps_per_epoch)
    total_steps = steps_per_epoch * epochs

    # Everything that creates variables goes inside the scope: under `TPUStrategy` a model built
    # outside it is not replicated across the cores, and the optimizer slots compile follows.
    with strategy.scope():
        lm: LanguageModel = BaseModel.build_lm(model_config)
        lm.summary()

        if hasattr(lm, "fit_counts"):
            raise ValueError(
                f"{type(lm).__name__} is fitted by counting, not gradient descent, so this script would only waste time on it. "
                f"Use `tensorflow_asr train_kenlm_lm` instead, which also writes lm_config.external_config."
            )

        logger.info(
            f"Training {type(lm).__name__} ({lm.count_params() / 1e6:.1f}M params) on the LM dataset "
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

    # No save here: `ModelCheckpoint` in `build_callbacks` has been writing the weights every epoch,
    # so a run that is interrupted still leaves a usable model instead of nothing.
    path = lm_weights_path(modeldir, "external")
    logger.info(f"Trained {type(lm).__name__} weights are at {path}")
    return path


if __name__ == "__main__":
    cli_util.run(main)
