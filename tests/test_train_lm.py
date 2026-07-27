"""
Tests for the language model training pipeline (`tensorflow_asr/scripts/train_lm.py`).

The part worth testing is the data plumbing, not the optimizer. Teacher forcing has to line up
exactly with what the beam search does at decoding time -- position `u` predicts token `u` from
everything before it, with blank standing in for start of sentence -- and the padding has to be
masked out of the loss, which cannot be inferred from the values because blank is both the pad
value and a legal token.
"""

import gzip

import numpy as np
import pytest

from tensorflow_asr import keras, tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.models.lm.lstm_language_model import LSTMLanguageModel
from tensorflow_asr.scripts.train_lm import TARGETS, text_line_tokens, to_training_pairs
from tensorflow_asr.tokenizers import CharTokenizer

LINES = ["THE QUICK BROWN FOX", "AB", "A LAZY DOG SLEEPS HERE"]


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None}))
    tok.make()
    return tok


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
