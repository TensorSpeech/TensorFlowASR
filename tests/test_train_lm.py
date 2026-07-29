"""
Tests for the language model training pipeline (`tensorflow_asr/scripts/train_lm.py`).

The part worth testing is the data plumbing, not the optimizer. Teacher forcing has to line up
exactly with what the beam search does at decoding time -- position `u` predicts token `u` from
everything before it, with blank standing in for start of sentence -- and the padding has to be
masked out of the loss, which cannot be inferred from the values because blank is both the pad
value and a legal token.
"""

import gzip
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorflow_asr import callbacks as asr_callbacks
from tensorflow_asr import keras, tf
from tensorflow_asr.configs import DatasetConfig, DecoderConfig
from tensorflow_asr.models.lm.lstm_language_model import LSTMLanguageModel
from tensorflow_asr.scripts.train_lm import (
    LR_SCHEDULES,
    TARGETS,
    MaskedSparseCategoricalCrossentropy,
    build_callbacks,
    build_optimizer,
    check_steps_per_epoch,
    text_line_tokens,
    to_training_pairs,
    transcript_token_dataset,
)
from tensorflow_asr.tokenizers import CharTokenizer

LINES = ["THE QUICK BROWN FOX", "AB", "A LAZY DOG SLEEPS HERE"]


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None}))
    tok.make()
    return tok


@pytest.fixture
def transcripts(tokenizer, tmp_path):
    """
    The args for `transcript_token_dataset`, backed by a transcript tsv.

    A real tsv rather than a stub, because the bug being guarded against lived in how the entries
    were handed to `tf.data`. No audio is read: the language model path stops at `read_entries`.
    """
    path = tmp_path / "transcripts.tsv"
    rows = "\n".join(f"/audio/{index}.flac\t1.0\t{line}" for index, line in enumerate(LINES))
    path.write_text(f"PATH\tDURATION\tTRANSCRIPT\n{rows}\n")
    return tokenizer, "generator", DatasetConfig({"data_paths": [str(path)], "enabled": True})


@pytest.fixture(params=["plain", "gzip"])
def corpus(request, tmp_path_factory):
    """A text corpus in both the forms OpenSLR ships: plain, and gzipped like the LibriSpeech one."""
    directory = tmp_path_factory.mktemp("corpus")
    if request.param == "gzip":
        path = directory / "lm-norm.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")
    else:
        path = directory / "lm-norm.txt"
        path.write_text("\n".join(LINES) + "\n")
    return str(path)


# --------------------------------------------------------------------------------------------
# reading the corpus
# --------------------------------------------------------------------------------------------


def test_reads_plain_and_gzipped_text(tokenizer, corpus):
    """`.gz` must be read directly -- the LibriSpeech LM corpus is several GB decompressed."""
    rows = [row.numpy() for row in text_line_tokens(tokenizer, corpus)]
    assert len(rows) == len(LINES)
    assert [len(r) for r in rows] == [len(line) for line in LINES], "one token per character"
    assert all(r.dtype == np.int32 for r in rows)


def test_tokens_match_the_tokenizer(tokenizer, corpus):
    """Indices must agree with the transducer's vocabulary, which is why the config's tokenizer is used."""
    first = next(iter(text_line_tokens(tokenizer, corpus))).numpy()
    np.testing.assert_array_equal(first, tokenizer.tokenize(LINES[0]).numpy())


def test_max_lines_caps_the_stream(tokenizer, corpus):
    assert len(list(text_line_tokens(tokenizer, corpus, max_lines=2))) == 2


# --------------------------------------------------------------------------------------------
# teacher forcing
# --------------------------------------------------------------------------------------------


def test_inputs_are_targets_shifted_by_one(tokenizer, corpus):
    pairs = to_training_pairs(text_line_tokens(tokenizer, corpus), blank=tokenizer.blank, batch_size=3, max_length=64)
    inputs, targets, weights = next(iter(pairs))

    assert np.all(inputs.numpy()[:, 0] == tokenizer.blank), "blank stands in for start of sentence"
    np.testing.assert_array_equal(inputs.numpy()[:, 1:], targets.numpy()[:, :-1])
    assert inputs.shape == targets.shape == weights.shape


def test_padding_is_masked_out_of_the_loss(tokenizer, corpus):
    """Blank is the pad value *and* a legal token, so the mask cannot be derived from the values."""
    pairs = to_training_pairs(text_line_tokens(tokenizer, corpus), blank=tokenizer.blank, batch_size=3, max_length=64)
    _, targets, weights = next(iter(pairs))

    lengths = [len(line) for line in LINES]
    assert weights.shape[1] == max(lengths), "padded to the longest in the batch, not to max_length"
    for row, length in zip(weights.numpy(), lengths):
        assert row[:length].sum() == length and row[length:].sum() == 0.0
    # and the padded positions really do hold blank, which is what makes the mask necessary
    assert np.all(targets.numpy()[weights.numpy() == 0] == tokenizer.blank)


def test_max_length_truncates(tokenizer, corpus):
    pairs = to_training_pairs(text_line_tokens(tokenizer, corpus), blank=tokenizer.blank, batch_size=3, max_length=5)
    inputs, _, _ = next(iter(pairs))
    assert inputs.shape[1] == 5


def test_empty_lines_are_dropped(tokenizer, tmp_path):
    path = tmp_path / "with-blanks.txt"
    path.write_text("HELLO\n\n\nWORLD\n")
    pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=8, max_length=64)
    _, targets, _ = next(iter(pairs))
    assert targets.shape[0] == 2, "an empty line carries no supervision"


# --------------------------------------------------------------------------------------------
# the loss actually learns from it
# --------------------------------------------------------------------------------------------


def test_training_reduces_loss_and_learns_the_corpus(tokenizer, tmp_path):
    """
    End to end on a corpus with exactly two possible openings, so what the LM should learn is
    checkable rather than merely "the number went down".
    """
    path = tmp_path / "c.txt"
    path.write_text("\n".join(["AB", "TX"] * 40) + "\n")
    pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=8, max_length=32)

    lm = LSTMLanguageModel(vocab_size=tokenizer.num_classes, embed_dim=16, units=32, nlayers=1)
    lm.make()
    lm.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        # `call` returns log-probabilities, and softmax(ln p) = p, so from_logits=True is a no-op
        # re-normalisation rather than a second softmax
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    history = lm.fit(pairs, epochs=20, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]

    probs = np.exp(lm.call_next(tf.constant([[tokenizer.blank]], tf.int32), lm.get_initial_state(1))[0].numpy()[0])
    starts = [int(tokenizer.tokenize(line).numpy()[0]) for line in ("AB", "TX")]
    assert probs[starts].sum() > 0.9, f"the two corpus openings should take nearly all the mass, got {probs[starts].sum():.3f}"
    assert probs[tokenizer.blank] < 0.05, "blank is only padding here; the mask should keep it out"


def test_targets_are_the_two_supported_names():
    assert TARGETS == ("external", "internal")


# --------------------------------------------------------------------------------------------
# the loss has to ignore padding in the denominator, not just the numerator
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("padding_fraction", [0.0, 0.25, 0.5, 0.75])
def test_loss_is_per_real_token_regardless_of_padding(padding_fraction):
    """
    Keras reduces with `sum_over_batch_size`, dividing by the element count, so padded positions
    deflate the stock loss even though they contribute nothing: at 50% padding it reports half the
    true cross entropy. That makes the number incomparable to any published perplexity and scales
    each batch's gradient by however much padding it happened to contain.
    """
    vocab, length = 100, 8
    real = int(length * (1 - padding_fraction))
    weights = tf.constant([[1.0] * real + [0.0] * (length - real)])
    # uniform logits, so the true per-token cross entropy is exactly ln(vocab)
    value = float(MaskedSparseCategoricalCrossentropy()(tf.constant([[1] * length]), tf.zeros([1, length, vocab]), sample_weight=weights))

    np.testing.assert_allclose(value, np.log(vocab), rtol=1e-5)


def test_loss_reaches_the_model_through_fit(tokenizer, corpus):
    """Overriding `__call__` bypasses Keras's reduction, so check `fit` really routes weights in."""
    pairs = to_training_pairs(text_line_tokens(tokenizer, corpus), blank=tokenizer.blank, batch_size=3, max_length=64)
    lm = LSTMLanguageModel(vocab_size=tokenizer.num_classes, embed_dim=8, units=16, nlayers=1)
    lm.make()
    lm.compile(optimizer=keras.optimizers.SGD(0.0), loss=MaskedSparseCategoricalCrossentropy())

    # a frozen model on a padded batch: the logged loss must be the per-token value, and the
    # untrained model is near uniform, so it should sit close to ln(V) rather than a fraction of it
    logged = lm.fit(pairs, epochs=1, verbose=0).history["loss"][0]
    assert 0.75 * np.log(tokenizer.num_classes) < logged < 1.25 * np.log(tokenizer.num_classes), (
        f"expected ~ln({tokenizer.num_classes})={np.log(tokenizer.num_classes):.2f} per token, got {logged:.2f}"
    )


# --------------------------------------------------------------------------------------------
# shuffling, clipping and the schedule
# --------------------------------------------------------------------------------------------


def test_shuffle_buffer_reorders_the_stream(tokenizer, tmp_path):
    """Both sources arrive in a fixed order, so without this the model sees one slice at a time."""
    path = tmp_path / "ordered.txt"
    path.write_text("\n".join(f"{'A' * (i % 20 + 1)}" for i in range(200)) + "\n")

    def first_lengths(shuffle_buffer):
        pairs = to_training_pairs(
            text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=8, max_length=64, shuffle_buffer=shuffle_buffer
        )
        _, targets, weights = next(iter(pairs))
        return list(weights.numpy().sum(axis=1))

    assert first_lengths(0) == sorted(first_lengths(0)), "unshuffled, the file order is preserved"
    assert any(first_lengths(200) != first_lengths(0) for _ in range(3)), "shuffling must change the order"


def test_repeat_lets_a_finite_dataset_fill_fixed_epochs(tokenizer, corpus):
    """`steps_per_epoch` on a finite dataset runs dry part way through unless it cycles."""
    pairs = to_training_pairs(text_line_tokens(tokenizer, corpus), blank=tokenizer.blank, batch_size=2, max_length=64, repeat=True)
    assert sum(1 for _ in pairs.take(20)) == 20, "a repeating dataset never runs out"

    finite = to_training_pairs(text_line_tokens(tokenizer, corpus), blank=tokenizer.blank, batch_size=2, max_length=64)
    assert sum(1 for _ in finite) == 2, "3 lines at batch size 2 is 2 batches and then it stops"


# --------------------------------------------------------------------------------------------
# the epoch length is given, not derived
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 0, -1])
def test_steps_per_epoch_is_required(bad):
    """Deriving it costs a walk over the whole corpus before the first step, so it is asked for."""
    with pytest.raises(ValueError, match="steps-per-epoch is required"):
        check_steps_per_epoch(bad)
    assert check_steps_per_epoch(1) == 1
    assert check_steps_per_epoch(7813) == 7813


def test_the_transcripts_can_be_read_twice(transcripts):
    """
    Regression: `transcript_tokens` is a one-shot generator, and wrapping it in `from_generator`
    made every read after the first yield nothing. Anything that inspects the dataset before
    training -- counting it, peeking at a batch -- would then train on an empty stream.
    """
    tokens = transcript_token_dataset(*transcripts)
    assert sum(1 for _ in tokens) == len(LINES)
    assert sum(1 for _ in tokens) == len(LINES), "reading it must not consume it"


def test_blank_lines_are_dropped(tokenizer, tmp_path):
    """They carry no supervision, so the pipeline yields nothing for them."""
    path = tmp_path / "gappy.txt"
    path.write_text("HELLO\n\n   \nWORLD\n")

    pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=1, max_length=64)
    assert sum(1 for _ in pairs) == 2, "4 lines in, but only 2 hold text"


# --------------------------------------------------------------------------------------------
# TPU needs one shape for every step
# --------------------------------------------------------------------------------------------


def test_static_shapes_for_xla(tokenizer, tmp_path):
    """
    XLA compiles per input shape. Padding each batch to its own longest sequence gives a new shape
    almost every step, which on TPU means recompiling instead of training.
    """
    path = tmp_path / "varied.txt"
    path.write_text("\n".join("A" * (i % 17 + 1) for i in range(40)) + "\n")

    def shapes(**kwargs):
        pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=4, max_length=32, **kwargs)
        return {tuple(t.shape) for t, _, _ in pairs}

    assert len(shapes()) > 1, "the default pipeline really does produce many shapes"
    assert shapes(padded_length=32, drop_remainder=True) == {(4, 32)}, "pinned, every batch is identical"


def test_drop_remainder_removes_the_short_batch(tokenizer, tmp_path):
    path = tmp_path / "five.txt"
    path.write_text("\n".join(f"LINE {i}" for i in range(5)) + "\n")

    def batch_rows(**kwargs):
        pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=4, max_length=32, **kwargs)
        return [int(w.shape[0]) for _, _, w in pairs]

    assert batch_rows() == [4, 1]
    assert batch_rows(drop_remainder=True) == [4], "the 5th sequence is skipped this pass"


DISTRIBUTED_LOSS_PROBE = """
import tensorflow as tf
tf.config.set_logical_device_configuration(
    tf.config.list_physical_devices("CPU")[0], [tf.config.LogicalDeviceConfiguration()] * 2
)
import numpy as np
from tensorflow_asr import keras
from tensorflow_asr.models.lm.lstm_language_model import LSTMLanguageModel
from tensorflow_asr.scripts.train_lm import MaskedSparseCategoricalCrossentropy

V, N, L = 30, 64, 8
x = np.random.randint(1, V, (N, L)).astype("int32")
w = np.ones((N, L), "float32")
ds = tf.data.Dataset.from_tensor_slices((x, x, w)).batch(8, drop_remainder=True).repeat()

strategy = tf.distribute.MirroredStrategy(["/cpu:0", "/cpu:1"])
with strategy.scope():
    lm = LSTMLanguageModel(vocab_size=V, embed_dim=8, units=16, nlayers=1)
    lm.make()
    # a frozen model, so the loss stays at the uniform baseline and only the reduction is measured
    lm.compile(optimizer=keras.optimizers.SGD(0.0), loss=MaskedSparseCategoricalCrossentropy())
    loss = lm.fit(ds, epochs=1, steps_per_epoch=2, verbose=0).history["loss"][0]

print(f"{strategy.num_replicas_in_sync} {loss} {np.log(V)}")
"""


def test_loss_does_not_scale_with_replica_count(tmp_path):
    """
    Keras sums what each replica's loss returns. Dividing by the local token count would make both
    the reported loss and the gradient scale with the replica count -- 8x on a TPU v3-8, silently
    multiplying the learning rate and making clipnorm bite eight times harder. The all-reduce in
    `MaskedSparseCategoricalCrossentropy` is what stops that, and nothing else here would catch it.

    Runs in a subprocess: virtual devices can only be configured before TensorFlow initialises its
    context, which any earlier test in the session will already have done.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(DISTRIBUTED_LOSS_PROBE)
    completed = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=900, check=False)

    if completed.returncode != 0:
        pytest.skip(f"could not run the distributed probe: {completed.stderr.strip().splitlines()[-1:]}")

    replicas, loss, uniform = (float(v) for v in completed.stdout.strip().splitlines()[-1].split())
    assert replicas == 2, "the probe needs two replicas to say anything"
    np.testing.assert_allclose(loss, uniform, rtol=0.02), "loss must be the per-token value, not replicas x it"


def test_padding_to_max_length_still_masks_the_loss(tokenizer, tmp_path):
    """Pinning the length adds a lot of padding, so the mask matters more, not less."""
    path = tmp_path / "short.txt"
    path.write_text("AB\nCD\n")
    pairs = to_training_pairs(
        text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=2, max_length=32, padded_length=32, drop_remainder=True
    )
    _, targets, weights = next(iter(pairs))

    assert targets.shape == (2, 32)
    assert weights.numpy().sum() == 4, "only the 4 real tokens are supervised, the other 60 slots are padding"
    # and the masked loss still reports a per-token value despite 94% padding
    value = float(MaskedSparseCategoricalCrossentropy()(targets, tf.zeros([2, 32, tokenizer.num_classes]), sample_weight=weights))
    np.testing.assert_allclose(value, np.log(tokenizer.num_classes), rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_one_epoch_of_counted_steps_is_exactly_one_pass(tokenizer, tmp_path, batch_size):
    """
    What makes `ceil(sequences / bs)` the right number to hand to --steps-per-epoch: `repeat` is
    applied after batching, so a cycle is exactly that many batches. Repeating the sequences
    instead would let batches straddle the seam and an "epoch" would drift out of step with the
    data.
    """
    path = tmp_path / "many.txt"
    path.write_text("\n".join(f"LINE {i}" for i in range(10)) + "\n")

    steps = math.ceil(10 / batch_size)
    pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=batch_size, max_length=64, repeat=True)

    # every sequence seen exactly once across one epoch's worth of steps
    rows = sum(int(w.numpy().shape[0]) for _, _, w in pairs.take(steps))
    assert rows == 10, f"one epoch of {steps} steps at bs={batch_size} covered {rows} sequences, expected 10"


def test_repeat_after_batching_keeps_the_partial_batch(tokenizer, tmp_path):
    """Sizes must be [4, 1] then [4, 1] again -- not merged into [4, 4, ...] across the cycle."""
    path = tmp_path / "five.txt"
    path.write_text("\n".join(f"LINE {i}" for i in range(5)) + "\n")
    pairs = to_training_pairs(text_line_tokens(tokenizer, str(path)), blank=tokenizer.blank, batch_size=4, max_length=64, repeat=True)

    assert [int(w.numpy().shape[0]) for _, _, w in pairs.take(4)] == [4, 1, 4, 1]


def test_clipping_is_applied_and_can_be_disabled():
    assert build_optimizer(1e-3, None, 100, clipnorm=1.0, lr_schedule="constant").clipnorm == 1.0
    assert build_optimizer(1e-3, None, 100, clipnorm=0, lr_schedule="constant").clipnorm is None


def rate_at(optimizer, step):
    """
    The rate the optimizer would actually use at `step`.

    Read through `optimizer.iterations` rather than off the schedule object: in Keras 3
    `optimizer.learning_rate` is the current *value*, so this checks the optimizer is really
    driving the schedule and not merely holding one.
    """
    optimizer.iterations.assign(step)
    return float(optimizer.learning_rate)


def test_cosine_schedule_warms_up_then_decays():
    optimizer = build_optimizer(1e-3, total_steps=1000, warmup_steps=100, clipnorm=1.0, lr_schedule="cosine")

    assert rate_at(optimizer, 0) < 1e-4, "starts near zero"
    np.testing.assert_allclose(rate_at(optimizer, 100), 1e-3, rtol=1e-4)  # peak at the end of warmup
    assert rate_at(optimizer, 550) < 1e-3, "decaying after the peak"
    assert rate_at(optimizer, 1000) < 1e-5, "and lands near zero at the step budget"


def test_warmup_is_capped_so_short_runs_are_not_all_warmup():
    optimizer = build_optimizer(1e-3, total_steps=200, warmup_steps=1000, clipnorm=1.0, lr_schedule="cosine")
    # warmup capped to total_steps // 10 = 20, so by then it is already at the peak
    np.testing.assert_allclose(rate_at(optimizer, 20), 1e-3, rtol=1e-4)


def test_cosine_without_a_step_budget_falls_back_to_a_constant_rate():
    """It cannot decay over an unknown horizon, so it holds the rate rather than inventing one."""
    schedule = build_optimizer(1e-3, total_steps=None, warmup_steps=100, clipnorm=1.0, lr_schedule="cosine").learning_rate
    np.testing.assert_allclose(float(schedule), 1e-3)


# --------------------------------------------------------------------------------------------
# checkpointing, so an interrupted run resumes
# --------------------------------------------------------------------------------------------


def test_no_checkpointing_without_a_handle(tmp_path):
    """The right default for a short run: uploading every epoch would cost more than restarting."""
    assert build_callbacks(str(tmp_path)) == []
    assert build_callbacks(None) == []


def test_handle_builds_a_kaggle_backup_callback(tmp_path):
    """
    Weights are only written to --output once `fit` returns, so a killed session otherwise loses
    everything. This callback checks the state into a Kaggle model after each epoch and pulls it
    back in `on_train_begin`.
    """
    callbacks = build_callbacks(str(tmp_path), kaggle_model_handle="owner/lm/keras/external")

    assert len(callbacks) == 1
    callback = callbacks[0]
    assert isinstance(callback, asr_callbacks.KaggleModelBackupAndRestore)
    assert isinstance(callback, keras.callbacks.BackupAndRestore), "it has to restore training state, not just upload files"
    config = callback.get_config()
    assert config["model_handle"] == "owner/lm/keras/external"
    assert config["save_freq"] == "epoch"
    # the local checkpoint lives under modeldir, which is what gets uploaded
    assert str(tmp_path) in callback.backup_dir


def test_handle_without_modeldir_is_rejected():
    """There is nowhere to write the checkpoint before uploading it."""
    with pytest.raises(ValueError, match="needs --modeldir"):
        build_callbacks(None, kaggle_model_handle="owner/lm/keras/external")
    with pytest.raises(ValueError, match="needs --modeldir"):
        build_callbacks("", kaggle_model_handle="owner/lm/keras/external")


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        build_optimizer(1e-3, 1000, 100, clipnorm=1.0, lr_schedule="triangular")
    assert LR_SCHEDULES == ("cosine", "constant")
