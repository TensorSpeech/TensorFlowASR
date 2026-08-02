"""
Tests for the language model training pipeline: the three trainers (train_external_lm,
train_internal_lm, train_kenlm_lm) and the KenLM corpus path on `LMDataset`.

The part worth testing is the data plumbing, not the optimizer. Teacher forcing has to line up
exactly with what the beam search does at decoding time -- position `u` predicts token `u` from
everything before it, with blank standing in for start of sentence -- and the padding has to be
masked out of the loss, which cannot be inferred from the values because blank is both the pad
value and a legal token.
"""

import gzip
import inspect
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from tensorflow_asr import callbacks as asr_callbacks
from tensorflow_asr import keras, tf
from tensorflow_asr.configs import DatasetConfig, DecoderConfig
from tensorflow_asr.datasets import LMDataset, get_lm
from tensorflow_asr.models.lm.lstm_language_model import LSTMLanguageModel
from tensorflow_asr.scripts.train_external_lm import (
    LR_SCHEDULES,
    MaskedSparseCategoricalCrossentropy,
    build_callbacks,
    build_optimizer,
    check_steps_per_epoch,
)
from tensorflow_asr.tokenizers import CharTokenizer

LINES = ["THE QUICK BROWN FOX", "AB", "A LAZY DOG SLEEPS HERE"]


# The pipeline lives in `datasets.LMDataset`. These two helpers keep the tests below pointed at its
# seams over one corpus file: `lm_tokens` is the tokenized `int32` stream (reading/tokenization),
# and `lm_pairs` is the batched `(inputs, targets, sample_weight)` from `create`, with the old
# `to_training_pairs` knobs mapped onto the dataset config. `padded_length` is gone -- a set
# `max_length` is itself the fixed padded length; left at 0, batches pad to their own longest.
def lm_tokens(tokenizer, text_path, **kwargs):
    return LMDataset(stage="train", tokenizer=tokenizer, data_paths=[str(text_path)], **kwargs)._token_dataset()


def lm_pairs(tokenizer, text_path, batch_size, *, max_length=0, shuffle_buffer=0, repeat=False, drop_remainder=False):
    dataset = LMDataset(
        stage="train",
        tokenizer=tokenizer,
        data_paths=[str(text_path)],
        max_length=max_length,
        shuffle=shuffle_buffer > 0,
        buffer_size=shuffle_buffer,
        drop_remainder=drop_remainder,
        indefinite=repeat,
    )
    return dataset.create(batch_size=batch_size)


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None}))
    tok.make()
    return tok


@pytest.fixture
def transcripts(tokenizer, tmp_path):
    """
    An `LMDataset` over a transcript tsv -- the internal-LM source, read as text.

    A real tsv rather than a stub, because the bug being guarded against lived in how the entries
    were handed to `tf.data`. No audio is read: the language model path only needs the text column.
    """
    path = tmp_path / "transcripts.tsv"
    rows = "\n".join(f"/audio/{index}.flac\t1.0\t{line}" for index, line in enumerate(LINES))
    path.write_text(f"PATH\tDURATION\tTRANSCRIPT\n{rows}\n")
    return get_lm(tokenizer=tokenizer, dataset_config=DatasetConfig({"data_paths": [str(path)], "enabled": True, "stage": "train"}))


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
    rows = [row.numpy() for row in lm_tokens(tokenizer, corpus)]
    assert len(rows) == len(LINES)
    assert [len(r) for r in rows] == [len(line) for line in LINES], "one token per character"
    assert all(r.dtype == np.int32 for r in rows)


def test_tokens_match_the_tokenizer(tokenizer, corpus):
    """Indices must agree with the transducer's vocabulary, which is why the config's tokenizer is used."""
    first = next(iter(lm_tokens(tokenizer, corpus))).numpy()
    np.testing.assert_array_equal(first, tokenizer.tokenize(LINES[0]).numpy())


def test_max_lines_caps_the_stream(tokenizer, corpus):
    assert len(list(lm_tokens(tokenizer, corpus, max_lines=2))) == 2


# --------------------------------------------------------------------------------------------
# LMDataset: mixing a transcript tsv with a text corpus, and the metadata it precomputes
# --------------------------------------------------------------------------------------------


def test_lmdataset_streams_tsv_and_text_together(tokenizer, tmp_path):
    """`data_paths` mixes an ASR transcript tsv with plain and gzipped text; all stream as one corpus."""
    tsv = tmp_path / "t.tsv"
    tsv.write_text("PATH\tDURATION\tTRANSCRIPT\n/a.flac\t1.0\tHELLO\n")
    txt = tmp_path / "c.txt"
    txt.write_text("WORLD\nAGAIN\n")
    gz = tmp_path / "c.txt.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write("ZIPPED\n")

    ds = LMDataset(stage="train", tokenizer=tokenizer, data_paths=[str(tsv), str(txt), str(gz)])
    assert list(ds.vocab_generator()) == ["HELLO", "WORLD", "AGAIN", "ZIPPED"], "header dropped, transcript column only"

    lengths = [int(tf.shape(t)[0]) for t in ds._token_dataset()]
    assert lengths == [len(w) for w in ("HELLO", "WORLD", "AGAIN", "ZIPPED")], "one token per character"
    assert lengths == [int(tf.shape(t)[0]) for t in ds._token_dataset()], "reading it must not consume it"


def test_lmdataset_metadata_round_trips_max_input_length(tokenizer, tmp_path):
    """`max_input_length` is the longest tokenised line, saved per stage and reloaded on construction."""
    txt = tmp_path / "c.txt"
    txt.write_text("HI\nA LONGER LINE\nMID\n")
    meta = tmp_path / "lm_metadata.json"

    ds = LMDataset(stage="train", tokenizer=tokenizer, data_paths=[str(txt)], metadata=str(meta))
    ds.compute_metadata()
    assert ds.max_input_length == len("A LONGER LINE"), "one token per character, spaces included"
    assert ds.num_entries == 3
    ds.save_metadata()

    reloaded = LMDataset(stage="train", tokenizer=tokenizer, data_paths=[str(txt)], metadata=str(meta))
    assert reloaded.max_input_length == len("A LONGER LINE"), "loaded from the metadata file, not recomputed"
    assert reloaded.num_entries == 3


def test_lmdataset_max_lines_caps_the_corpus(tokenizer, tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("\n".join(f"LINE{i}" for i in range(10)) + "\n")
    ds = LMDataset(stage="train", tokenizer=tokenizer, data_paths=[str(txt)], max_lines=3)
    assert sum(1 for _ in ds._token_dataset()) == 3
    assert list(ds.vocab_generator()) == ["LINE0", "LINE1", "LINE2"]


# --------------------------------------------------------------------------------------------
# teacher forcing
# --------------------------------------------------------------------------------------------


def test_inputs_are_targets_shifted_by_one(tokenizer, corpus):
    pairs = lm_pairs(tokenizer, corpus, 3, max_length=64)
    inputs, targets, weights = next(iter(pairs))

    assert np.all(inputs.numpy()[:, 0] == tokenizer.blank), "blank stands in for start of sentence"
    np.testing.assert_array_equal(inputs.numpy()[:, 1:], targets.numpy()[:, :-1])
    assert inputs.shape == targets.shape == weights.shape


def test_padding_is_masked_out_of_the_loss(tokenizer, corpus):
    """Blank is the pad value *and* a legal token, so the mask cannot be derived from the values."""
    # max_length left at 0 so the batch pads to its own longest sequence, which is what this checks.
    pairs = lm_pairs(tokenizer, corpus, 3)
    _, targets, weights = next(iter(pairs))

    lengths = [len(line) for line in LINES]
    assert weights.shape[1] == max(lengths), "padded to the longest in the batch"
    for row, length in zip(weights.numpy(), lengths):
        assert row[:length].sum() == length and row[length:].sum() == 0.0
    # and the padded positions really do hold blank, which is what makes the mask necessary
    assert np.all(targets.numpy()[weights.numpy() == 0] == tokenizer.blank)


def test_max_length_truncates(tokenizer, corpus):
    pairs = lm_pairs(tokenizer, corpus, 3, max_length=5)
    inputs, _, _ = next(iter(pairs))
    assert inputs.shape[1] == 5


def test_empty_lines_are_dropped(tokenizer, tmp_path):
    path = tmp_path / "with-blanks.txt"
    path.write_text("HELLO\n\n\nWORLD\n")
    pairs = lm_pairs(tokenizer, str(path), 8, max_length=64)
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
    pairs = lm_pairs(tokenizer, str(path), 8, max_length=32)

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


def test_each_trainer_targets_one_lm_config_key():
    """
    The `--target` flag is gone: which language model gets built is now the script you run. The
    external key has two trainers because a neural LM and an n-gram are alternatives for it.
    """
    from tensorflow_asr.scripts import train_external_lm, train_internal_lm, train_kenlm_lm

    assert "external_config" in inspect.getsource(train_external_lm.main)
    assert "external_config" in inspect.getsource(train_kenlm_lm.main)
    assert "internal_config" in inspect.getsource(train_internal_lm.main)


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


@pytest.mark.parametrize("bad", [-np.inf, np.inf, np.nan])
def test_padding_cannot_poison_the_loss(bad):
    """
    Regression: the mask must *drop* padded positions, not multiply them by zero.

    `inf * 0` and `nan * 0` are both `nan`, so applying the mask by multiplication let a single
    non-finite value at a position nobody is supervising take out the whole loss -- and from there
    the gradients, the weights, and every subsequent epoch. Padded positions are real forward passes
    over pad tokens, so they are exactly where such a value appears unnoticed.
    """
    vocab, length = 6, 4
    y_true = tf.zeros([1, length], tf.int32)
    weights = tf.constant([[1.0, 1.0, 0.0, 0.0]])  # last two positions are padding

    logits = np.zeros([1, length, vocab], "float32")
    logits[0, 3, 0] = bad  # non-finite, but only where the mask is 0

    value = float(MaskedSparseCategoricalCrossentropy()(y_true, tf.constant(logits), sample_weight=weights))
    np.testing.assert_allclose(value, np.log(vocab), rtol=1e-5)


def test_a_non_finite_real_token_still_shows_up():
    """The guard must not hide a genuinely diverged model, only ignore what it was told to ignore."""
    vocab, length = 6, 4
    logits = np.zeros([1, length, vocab], "float32")
    logits[0, 0, 0] = np.nan  # a *supervised* position

    value = float(
        MaskedSparseCategoricalCrossentropy()(tf.zeros([1, length], tf.int32), tf.constant(logits), sample_weight=tf.constant([[1.0, 1.0, 0.0, 0.0]]))
    )
    assert np.isnan(value), "divergence at a real token must still be reported, for TerminateOnNaN to catch"


def test_non_finite_gradients_never_reach_the_weights():
    """
    Regression: one non-finite gradient used to be fatal and permanent.

    There is no `LossScaleOptimizer` at a float32 policy to skip the step, and `clipnorm` cannot
    help -- the global norm of a vector containing NaN is NaN, so the clipped gradient is NaN too.
    """
    for clipnorm in (1.0, 0):
        optimizer = build_optimizer(1e-3, None, 100, clipnorm=clipnorm, lr_schedule="constant")
        variable = keras.Variable([1.0, 2.0, 3.0], name="w")
        optimizer.build([variable])
        optimizer.apply([tf.constant([np.nan, np.inf, 0.5], tf.float32)], [variable])

        values = np.asarray(variable)
        assert np.isfinite(values).all(), f"clipnorm={clipnorm} let a non-finite gradient through: {values}"
        assert values[0] == 1.0 and values[1] == 2.0, "the non-finite entries must be dropped, not applied"
        assert values[2] != 3.0, "the finite entry must still train"


def test_one_bad_gradient_cannot_poison_other_variables():
    """
    The sanitising has to happen *before* the clipping, which global clipping makes load-bearing.

    A global norm is one scalar over every gradient at once, so a single NaN anywhere makes that
    norm NaN and every variable is scaled by NaN -- turning one bad tensor into a dead model. That
    is strictly worse than the per-variable clipping this replaced, and is only safe because
    `NanSafeAdam.apply` zeroes the non-finite entries before delegating.
    """
    optimizer = build_optimizer(1e-3, None, 100, clipnorm=1.0, lr_schedule="constant")
    a = keras.Variable([1.0, 2.0], name="a")
    b = keras.Variable([3.0, 4.0], name="b")  # entirely healthy
    c = keras.Variable([5.0, 6.0], name="c")
    optimizer.build([a, b, c])

    optimizer.apply([tf.constant([np.nan, 0.0]), tf.constant([0.5, 0.5]), tf.constant([np.inf, 0.2])], [a, b, c])

    for v in (a, b, c):
        assert np.isfinite(np.asarray(v)).all(), f"{v.name} was poisoned across the global norm: {np.asarray(v)}"
    assert not np.allclose(np.asarray(b), [3.0, 4.0]), "the healthy variable must still train"


def test_loss_graph_has_no_device_bound_assertion():
    """
    Regression: the loss must not put an `Assert` in the graph.

    `sparse_categorical_crossentropy` reaches `tf.nn.sparse_softmax_cross_entropy_with_logits`,
    which adds a runtime shape check whenever the static shapes are not fully known -- every GPU
    batch, since those pad to the longest sequence in the batch rather than to `max_length`. The
    check is an `Assert`, `Assert` has no GPU kernel on any backend, and `MirroredStrategy` pins
    every op to the device, so it failed the run outright:

        Cannot assign a device for operation .../SparseSoftmaxCrossEntropyWithLogits/assert_equal_1
        ... no supported kernel for GPU devices is available

    Checked on the graph rather than by running on a GPU, so it holds in CPU-only CI.
    """
    vocab = 12
    signature = [
        tf.TensorSpec([None, None], tf.int32),
        tf.TensorSpec([None, None, vocab], tf.float32),
        tf.TensorSpec([None, None], tf.float32),
    ]
    graph = tf.function(MaskedSparseCategoricalCrossentropy(), input_signature=signature).get_concrete_function().graph
    assertions = sorted({op.type for op in graph.get_operations() if "Assert" in op.type})

    assert not assertions, f"these ops have no GPU kernel and would fail under MirroredStrategy: {assertions}"


def test_loss_reaches_the_model_through_fit(tokenizer, corpus):
    """Overriding `__call__` bypasses Keras's reduction, so check `fit` really routes weights in."""
    pairs = lm_pairs(tokenizer, corpus, 3, max_length=64)
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
        pairs = lm_pairs(tokenizer, str(path), 8, max_length=64, shuffle_buffer=shuffle_buffer)
        _, _, weights = next(iter(pairs))
        return list(weights.numpy().sum(axis=1))

    assert first_lengths(0) == sorted(first_lengths(0)), "unshuffled, the file order is preserved"
    assert any(first_lengths(200) != first_lengths(0) for _ in range(3)), "shuffling must change the order"


def test_repeat_lets_a_finite_dataset_fill_fixed_epochs(tokenizer, corpus):
    """`steps_per_epoch` on a finite dataset runs dry part way through unless it cycles."""
    pairs = lm_pairs(tokenizer, corpus, 2, max_length=64, repeat=True)
    assert sum(1 for _ in pairs.take(20)) == 20, "a repeating dataset never runs out"

    finite = lm_pairs(tokenizer, corpus, 2, max_length=64)
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
    Regression: the tokenized stream must be replayable. A one-shot generator wrapped in
    `from_generator` yields nothing after the first walk, so anything that inspects the dataset
    before training -- counting it, peeking at a batch -- would leave training with an empty stream.
    `from_tensor_slices` (for a tsv) and `TextLineDataset` both replay.
    """
    tokens = transcripts._token_dataset()
    assert sum(1 for _ in tokens) == len(LINES)
    assert sum(1 for _ in tokens) == len(LINES), "reading it must not consume it"


def test_blank_lines_are_dropped(tokenizer, tmp_path):
    """They carry no supervision, so the pipeline yields nothing for them."""
    path = tmp_path / "gappy.txt"
    path.write_text("HELLO\n\n   \nWORLD\n")

    pairs = lm_pairs(tokenizer, str(path), 1, max_length=64)
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
        return {tuple(t.shape) for t, _, _ in lm_pairs(tokenizer, str(path), 4, **kwargs)}

    assert len(shapes()) > 1, "left dynamic (max_length=0), every batch pads to its own longest -- many shapes"
    assert shapes(max_length=32, drop_remainder=True) == {(4, 32)}, "pinned by max_length, every batch is identical"


def test_drop_remainder_removes_the_short_batch(tokenizer, tmp_path):
    path = tmp_path / "five.txt"
    path.write_text("\n".join(f"LINE {i}" for i in range(5)) + "\n")

    def batch_rows(**kwargs):
        return [int(w.shape[0]) for _, _, w in lm_pairs(tokenizer, str(path), 4, max_length=32, **kwargs)]

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
from tensorflow_asr.scripts.train_external_lm import MaskedSparseCategoricalCrossentropy

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
    pairs = lm_pairs(tokenizer, str(path), 2, max_length=32, drop_remainder=True)
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
    pairs = lm_pairs(tokenizer, str(path), batch_size, max_length=64, repeat=True)

    # every sequence seen exactly once across one epoch's worth of steps
    rows = sum(int(w.numpy().shape[0]) for _, _, w in pairs.take(steps))
    assert rows == 10, f"one epoch of {steps} steps at bs={batch_size} covered {rows} sequences, expected 10"


def test_repeat_after_batching_keeps_the_partial_batch(tokenizer, tmp_path):
    """Sizes must be [4, 1] then [4, 1] again -- not merged into [4, 4, ...] across the cycle."""
    path = tmp_path / "five.txt"
    path.write_text("\n".join(f"LINE {i}" for i in range(5)) + "\n")
    pairs = lm_pairs(tokenizer, str(path), 4, max_length=64, repeat=True)

    assert [int(w.numpy().shape[0]) for _, _, w in pairs.take(4)] == [4, 1, 4, 1]


def test_clipping_is_applied_and_can_be_disabled():
    assert build_optimizer(1e-3, None, 100, clipnorm=1.0, lr_schedule="constant").global_clipnorm == 1.0
    assert build_optimizer(1e-3, None, 100, clipnorm=0, lr_schedule="constant").global_clipnorm is None


def test_clipping_is_global_not_per_variable():
    """
    Regression: `clipnorm` and `global_clipnorm` are not interchangeable.

    Keras' `clipnorm` rescales each weight tensor independently, so on the spikes clipping exists
    for it shrinks whichever tensors blew up and leaves the others, rotating the update away from
    the gradient. Pascanu et al. (2013) specify one scalar over the whole gradient for recurrent
    nets precisely so the direction survives.
    """
    optimizer = build_optimizer(1e-3, None, 100, clipnorm=1.0, lr_schedule="constant")
    assert optimizer.clipnorm is None, "per-variable clipping must stay off"
    assert optimizer.global_clipnorm == 1.0

    # one tensor spikes hard, another does not: per-variable clipping would change their ratio
    grads = [tf.constant([40.0, 0.0]), tf.constant([0.0, 1.0])]
    scale = 1.0 / float(tf.linalg.global_norm(grads))
    clipped = [g * min(1.0, scale) for g in grads]

    flat_raw = tf.concat([tf.reshape(g, [-1]) for g in grads], 0)
    flat_clipped = tf.concat([tf.reshape(g, [-1]) for g in clipped], 0)
    cosine = float(tf.reduce_sum(flat_raw * flat_clipped) / (tf.norm(flat_raw) * tf.norm(flat_clipped)))
    np.testing.assert_allclose(cosine, 1.0, atol=1e-6)  # direction exactly preserved
    np.testing.assert_allclose(float(tf.linalg.global_norm(clipped)), 1.0, rtol=1e-6)


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


def test_weights_are_checkpointed_every_epoch_without_a_handle(tmp_path):
    """
    There is no save after `fit` returns, so the checkpoint is the only thing that writes the model.

    Training a language model on a real corpus runs long enough that being stopped partway through
    is the normal case, and it has to leave a usable model from the last completed epoch rather than
    nothing. The Kaggle backup is a different thing -- it carries optimizer state so a run can
    *resume* -- and stays optional.
    """
    callbacks = build_callbacks(str(tmp_path))
    kinds = [type(c) for c in callbacks]
    assert asr_callbacks.TerminateOnNaN in kinds
    checkpoints = [c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint)]
    assert len(checkpoints) == 1, f"expected exactly one ModelCheckpoint, got {kinds}"
    assert str(checkpoints[0].filepath) == os.path.join(str(tmp_path), "lm", "external.weights.h5")
    assert checkpoints[0].save_weights_only is True


def test_nan_always_terminates_the_run(tmp_path):
    """
    A NaN loss is unrecoverable here, so the run must stop rather than burn the session.

    Nothing downstream can undo it: a float32 policy attaches no `LossScaleOptimizer` to skip the
    step, so once NaN reaches the weights every later forward pass is NaN -- across the epoch
    boundary, and into the checkpoint. This callback is therefore not optional the way the Kaggle
    backup is, and must be present with or without a handle.
    """
    plain = build_callbacks(str(tmp_path))
    with_handle = build_callbacks(str(tmp_path), kaggle_model_handle="owner/lm/keras/external")
    for callbacks in (plain, with_handle):
        assert any(isinstance(c, keras.callbacks.TerminateOnNaN) for c in callbacks)


def test_handle_builds_a_kaggle_backup_callback(tmp_path):
    """
    Weights are only written to --output once `fit` returns, so a killed session otherwise loses
    everything. This callback checks the state into a Kaggle model after each epoch and pulls it
    back in `on_train_begin`.
    """
    callbacks = build_callbacks(str(tmp_path), kaggle_model_handle="owner/lm/keras/external")

    backups = [c for c in callbacks if isinstance(c, asr_callbacks.KaggleModelBackupAndRestore)]
    assert len(backups) == 1
    callback = backups[0]
    assert isinstance(callback, keras.callbacks.BackupAndRestore), "it has to restore training state, not just upload files"
    config = callback.get_config()
    assert config["model_handle"] == "owner/lm/keras/external"
    assert config["save_freq"] == "epoch"
    # the local checkpoint lives under modeldir, which is what gets uploaded
    assert str(tmp_path) in callback.backup_dir


def test_callbacks_need_a_modeldir():
    """
    Every run writes its weights now, so there is always somewhere they have to go -- with or
    without a Kaggle handle.
    """
    for handle in (None, "owner/lm/keras/external"):
        for modeldir in (None, ""):
            with pytest.raises(ValueError, match="--modeldir is required"):
                build_callbacks(modeldir, kaggle_model_handle=handle)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        build_optimizer(1e-3, 1000, 100, clipnorm=1.0, lr_schedule="triangular")
    assert LR_SCHEDULES == ("cosine", "constant")


# --------------------------------------------------------------------------------------------
# the KenLM path: LMDataset.write_token_ids / create_arpa
# --------------------------------------------------------------------------------------------


STUB_LMPLZ = '''#!/usr/bin/env python3
"""Stand-in for KenLM lmplz: reads token-id text on stdin, writes a minimal valid ARPA."""
import sys
from collections import defaultdict

order = int(sys.argv[sys.argv.index("-o") + 1]) if "-o" in sys.argv else 3
uni, bi = defaultdict(int), defaultdict(int)
for line in sys.stdin:
    toks = line.split()
    if not toks:
        continue
    for t in toks:
        uni[t] += 1
    for a, b in zip(["<s>"] + toks, toks):
        bi[(a, b)] += 1
total = sum(uni.values()) + len(bi)
out = ["\\\\data\\\\", f"ngram 1={len(uni) + 2}", f"ngram 2={len(bi)}", ""]
out.append("\\\\1-grams:")
out.append("-99\\t<s>\\t-0.3")
out.append("-1.5\\t<unk>")
for t, c in uni.items():
    out.append(f"{-2.0:.4f}\\t{t}\\t-0.3")
out.append("")
out.append("\\\\2-grams:")
for (a, b), c in bi.items():
    out.append(f"{-1.0:.4f}\\t{a} {b}")
out.append("")
out.append("\\\\end\\\\")
sys.stdout.write("\\n".join(out) + "\\n")
'''


@pytest.fixture
def stub_lmplz(tmp_path):
    """A fake `lmplz` on PATH, so the subprocess plumbing is exercised without building KenLM."""
    path = tmp_path / "lmplz"
    path.write_text(STUB_LMPLZ)
    path.chmod(0o755)
    return str(path)


def test_write_token_ids_writes_what_lmplz_expects(transcripts, tmp_path):
    """One sentence per line, tokens as space-separated integer ids -- lmplz has no other contract."""
    path = tmp_path / "corpus.ids.txt"
    lines, tokens = transcripts.write_token_ids(str(path))
    assert lines == len(LINES) and tokens > 0

    written = path.read_text().strip().split("\n")
    assert len(written) == lines
    for row in written:
        ids = row.split()
        assert ids, "an empty line would read as a sentence and skew the counts"
        assert all(i.isdigit() for i in ids), f"non-integer token in {row!r}"


def test_write_token_ids_gzips_on_a_gz_suffix(transcripts, tmp_path):
    path = tmp_path / "corpus.ids.txt.gz"
    lines, _ = transcripts.write_token_ids(str(path))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert len(handle.read().strip().split("\n")) == lines


def test_write_token_ids_honours_max_lines(transcripts, tmp_path):
    path = tmp_path / "short.ids.txt"
    lines, _ = transcripts.write_token_ids(str(path), max_lines=2)
    assert lines == 2 and len(path.read_text().strip().split("\n")) == 2


def test_create_arpa_says_how_to_get_lmplz_when_it_is_missing(transcripts, tmp_path):
    with pytest.raises(FileNotFoundError, match="install_kenlm.sh"):
        transcripts.create_arpa(arpa_path=str(tmp_path / "lm.arpa"), lmplz="definitely-not-a-real-binary")


def test_create_arpa_builds_a_loadable_model(transcripts, tokenizer, stub_lmplz, tmp_path):
    """The whole KenLM path: tokenize, run lmplz, and convert what it wrote into the arc tensors."""
    from tensorflow_asr.models.lm.ngram_language_model import NGramLanguageModel

    arpa = transcripts.create_arpa(arpa_path=str(tmp_path / "lm.arpa"), order=2, lmplz=stub_lmplz)
    assert os.path.isfile(arpa)

    lm = NGramLanguageModel(vocab_size=tokenizer.num_classes, blank=tokenizer.blank, order=2, max_arcs=8192, max_states=4096)
    stats = lm.load_arpa(arpa)
    assert stats["source"] == "arpa" and stats["arcs"] > 0

    scores, _ = lm.call_next(tf.constant([[1]], tf.int32), lm.get_initial_state(1))
    labels = np.delete(np.exp(scores.numpy().astype(np.float64)), tokenizer.blank)
    np.testing.assert_allclose(labels.sum(), 1.0, atol=1e-5)


def test_create_arpa_reuses_the_token_ids_it_already_wrote(transcripts, stub_lmplz, tmp_path):
    """Tokenising is the slow half, so a second run at another order must not repeat it."""
    text = tmp_path / "corpus.ids.txt"
    transcripts.create_arpa(arpa_path=str(tmp_path / "a.arpa"), text_path=str(text), order=2, lmplz=stub_lmplz)
    stamp = text.stat().st_mtime_ns
    marker = text.read_text()

    transcripts.create_arpa(arpa_path=str(tmp_path / "b.arpa"), text_path=str(text), order=2, lmplz=stub_lmplz)
    assert text.stat().st_mtime_ns == stamp, "the token ids were rewritten instead of reused"
    assert text.read_text() == marker


def test_create_arpa_reports_a_failing_lmplz(transcripts, tmp_path):
    failing = tmp_path / "failing-lmplz"
    failing.write_text("#!/usr/bin/env bash\nexit 1\n")
    failing.chmod(0o755)
    with pytest.raises(RuntimeError, match="discount_fallback"):
        transcripts.create_arpa(arpa_path=str(tmp_path / "lm.arpa"), lmplz=str(failing))


def test_create_arpa_overwrites_the_token_ids_on_request(transcripts, stub_lmplz, tmp_path):
    """
    Reuse is right between runs that only change pruning, and wrong the moment the ids would differ
    -- a changed corpus, a changed tokenizer. A stale file is still a valid file, so nothing detects
    that and the flag is the only way out.
    """
    text = tmp_path / "corpus.ids.txt"
    transcripts.create_arpa(arpa_path=str(tmp_path / "a.arpa"), text_path=str(text), order=2, lmplz=stub_lmplz)
    text.write_text("1 2 3\n")  # stand in for a stale file from an older corpus

    transcripts.create_arpa(arpa_path=str(tmp_path / "b.arpa"), text_path=str(text), order=2, lmplz=stub_lmplz)
    assert text.read_text() == "1 2 3\n", "the default must still reuse whatever is there"

    transcripts.create_arpa(
        arpa_path=str(tmp_path / "c.arpa"), text_path=str(text), order=2, lmplz=stub_lmplz, overwrite_text=True
    )
    rebuilt = text.read_text()
    assert rebuilt != "1 2 3\n", "overwrite_text must re-tokenise rather than reuse"
    assert len(rebuilt.strip().split("\n")) == len(LINES)


def test_create_arpa_overwrite_works_when_there_is_nothing_to_overwrite(transcripts, stub_lmplz, tmp_path):
    """The flag must not require the file to exist -- a first run with it set is perfectly normal."""
    text = tmp_path / "fresh.ids.txt"
    assert not text.exists()
    transcripts.create_arpa(arpa_path=str(tmp_path / "lm.arpa"), text_path=str(text), order=2, lmplz=stub_lmplz, overwrite_text=True)
    assert len(text.read_text().strip().split("\n")) == len(LINES)
