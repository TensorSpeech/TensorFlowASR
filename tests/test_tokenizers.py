"""
Round-trip tests for the three tokenizers against the configs shipped in `examples/`.

Previously these pointed at `examples/configs/librispeech/...`, which moved to
`examples/datasets/librispeech/...`, so every case died on a missing file. They also never called
`make()` (so the tokenizers were uninitialised) and asserted nothing -- everything was printed.
"""

import os

import pytest

from tensorflow_asr import tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.tokenizers import CharTokenizer, SentencePieceTokenizer, WordPieceTokenizer
from tensorflow_asr.utils import file_util

file_util.ENABLE_PATH_PREPROCESS = False

REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_ROOT = os.path.join(REPO_ROOT, "examples", "datasets", "librispeech")

TEXT = (
    "i'm good but it would have broken down after ten miles of that hard trail dawn came while they "
    "wound over the crest of the range and with the sun in their faces they took the downgrade it "
    "was well into the morning before nash reached logan"
)

TOKENIZERS = [
    ("characters/char.yml.j2", CharTokenizer),
    ("wordpiece/wp.yml.j2", WordPieceTokenizer),
    ("wordpiece/wp_whitespace.yml.j2", WordPieceTokenizer),
    ("sentencepiece/sp.yml.j2", SentencePieceTokenizer),
    ("sentencepiece/sp.256.yml.j2", SentencePieceTokenizer),
]


def build_tokenizer(relative_config_path, tokenizer_class):
    config = file_util.load_yaml(os.path.join(CONFIG_ROOT, relative_config_path), repodir=REPO_ROOT)
    tokenizer = tokenizer_class(decoder_config=DecoderConfig(config["decoder_config"]))
    tokenizer.make()
    return tokenizer


@pytest.fixture(params=TOKENIZERS, ids=[path for path, _ in TOKENIZERS])
def tokenizer(request):
    return build_tokenizer(*request.param)


def test_tokenizer_is_initialized(tokenizer):
    assert tokenizer.initialized
    assert tokenizer.num_classes > 0
    assert 0 <= tokenizer.blank < tokenizer.num_classes


def test_tokenize_produces_in_range_indices(tokenizer):
    indices = tokenizer.tokenize(TEXT)
    assert indices.dtype == tf.int32
    assert int(tf.size(indices)) > 0
    assert bool(tf.reduce_all(indices >= 0))
    assert bool(tf.reduce_all(indices < tokenizer.num_classes))


def test_detokenize_round_trip(tokenizer):
    indices = tokenizer.tokenize(TEXT)
    batch = tf.stack([indices, indices], axis=0)
    transcripts = tokenizer.detokenize(batch)

    assert transcripts.shape == (2,)
    decoded = [t.decode("utf-8") for t in transcripts.numpy()]
    assert decoded[0] == decoded[1], "identical rows decoded differently"
    assert decoded[0] == TEXT


def test_trailing_blanks_are_ignored(tokenizer):
    """Decoders emit blank-padded token arrays, so padding must not change the transcript."""
    indices = tokenizer.tokenize(TEXT)
    padded = tf.concat([indices, tf.fill([8], tokenizer.blank)], axis=0)

    unpadded_text = tokenizer.detokenize(tf.stack([indices], axis=0)).numpy()[0].decode("utf-8")
    padded_text = tokenizer.detokenize(tf.stack([padded], axis=0)).numpy()[0].decode("utf-8")
    assert padded_text == unpadded_text


def test_detokenize_unicode_points(tokenizer):
    """The tflite path returns code points rather than strings; they must agree with detokenize."""
    indices = tokenizer.tokenize(TEXT)
    upoints = tokenizer.detokenize_unicode_points(indices)

    assert upoints.dtype == tf.int32
    decoded = "".join(chr(c) for c in upoints.numpy())
    assert decoded == tokenizer.detokenize(tf.stack([indices], axis=0)).numpy()[0].decode("utf-8")


def test_tokenizers_disagree_on_segmentation():
    """A sanity check that the configs really select different tokenizers."""
    counts = {}
    for relative_path, tokenizer_class in TOKENIZERS:
        tokenizer = build_tokenizer(relative_path, tokenizer_class)
        counts[relative_path] = int(tf.size(tokenizer.tokenize(TEXT)))

    # characters must be the most granular segmentation of the same text
    assert counts["characters/char.yml.j2"] == max(counts.values())
    assert len(set(counts.values())) > 1, f"every tokenizer produced the same token count: {counts}"
