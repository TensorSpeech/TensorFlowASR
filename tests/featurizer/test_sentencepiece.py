"""
Tests for `SentencePieceTokenizer` and the dataset pipeline that feeds it.

The previous version of this file targeted an API generation that no longer exists
(`SpeechFeaturizer`, `SubwordFeaturizer`, `Tokenizer.load_from_file`, `iextract`,
`ASRSliceTestDataset`, and `ASRSliceDataset(speech_featurizer=..., text_featurizer=...)`), and read
audio from hard-coded `/data/datasets/LibriSpeech` paths. It is rewritten here against the current
API and made self-contained: the sentencepiece model ships in `examples/`, and the audio is the
`tests/test.flac` fixture, so nothing depends on a corpus being mounted.
"""

import os

import pytest
import sentencepiece as spm

from tensorflow_asr import tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.datasets import ASRSliceDataset
from tensorflow_asr.tokenizers import SentencePieceTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SENTENCEPIECE_MODEL = os.path.join(REPO_ROOT, "examples", "datasets", "librispeech", "sentencepiece", "train_8000&960.model")
AUDIO_FIXTURE = os.path.join(REPO_ROOT, "tests", "test.flac")
TRANSCRIPT = "this is a test"


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = SentencePieceTokenizer(DecoderConfig({"type": "sentencepiece", "vocabulary": SENTENCEPIECE_MODEL, "blank_index": 0}))
    tokenizer.make()
    return tokenizer


@pytest.fixture
def transcripts_tsv(tmp_path):
    """A one-line manifest in the `PATH \\t DURATION \\t TRANSCRIPT` format the datasets expect."""
    path = tmp_path / "transcripts.tsv"
    path.write_text(f"PATH\tDURATION\tTRANSCRIPT\n{AUDIO_FIXTURE}\t1.0\t{TRANSCRIPT}\n")
    return str(path)


def test_sentencepiece_model_round_trip():
    """Raw sentencepiece behaviour, independent of any TensorFlowASR wrapper."""
    processor = spm.SentencePieceProcessor()
    processor.load(SENTENCEPIECE_MODEL)

    assert processor.decode_ids(processor.encode_as_ids(TRANSCRIPT)) == TRANSCRIPT
    assert processor.decode_pieces(processor.encode_as_pieces(TRANSCRIPT)) == TRANSCRIPT


def test_tokenizer_round_trip(tokenizer):
    assert tokenizer.initialized
    assert tokenizer.num_classes == 8000
    assert tokenizer.blank == 0

    indices = tokenizer.tokenize(tf.constant(TRANSCRIPT))
    assert indices.dtype == tf.int32
    assert int(tf.size(indices)) > 0

    decoded = tokenizer.detokenize(tf.reshape(indices, [1, -1])).numpy()[0].decode("utf-8")
    assert decoded == TRANSCRIPT


def test_tokenizer_matches_raw_sentencepiece(tokenizer):
    processor = spm.SentencePieceProcessor()
    processor.load(SENTENCEPIECE_MODEL)
    expected = processor.encode_as_ids(TRANSCRIPT)
    assert tokenizer.tokenize(tf.constant(TRANSCRIPT)).numpy().tolist() == expected


def test_dataset_produces_usable_batches(tokenizer, transcripts_tsv):
    dataset = ASRSliceDataset(
        stage="train",
        tokenizer=tokenizer,
        data_paths=[transcripts_tsv],
        shuffle=False,
        indefinite=False,
        drop_remainder=False,
    )
    inputs, labels = next(iter(dataset.create(1)))

    assert inputs.inputs.shape[0] == 1
    assert int(inputs.inputs_length[0]) == inputs.inputs.shape[1]
    assert int(labels.labels_length[0]) == int(tf.size(tokenizer.tokenize(tf.constant(TRANSCRIPT))))
    # the transducer prediction stream is the labels shifted by one and prefixed with blank
    assert int(inputs.predictions_length[0]) == int(labels.labels_length[0]) + 1
    assert int(inputs.predictions[0, 0]) == tokenizer.blank


def test_dataset_labels_detokenize_to_the_transcript(tokenizer, transcripts_tsv):
    """End to end: manifest -> tokenized labels -> back to the original text."""
    dataset = ASRSliceDataset(
        stage="train",
        tokenizer=tokenizer,
        data_paths=[transcripts_tsv],
        shuffle=False,
        indefinite=False,
        drop_remainder=False,
    )
    _, labels = next(iter(dataset.create(1)))
    decoded = tokenizer.detokenize(labels.labels).numpy()[0].decode("utf-8")
    assert decoded == TRANSCRIPT
