# pylint: disable=line-too-long
import os

from tensorflow_asr import tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.tokenizers import CharTokenizer, SentencePieceTokenizer, WordPieceTokenizer
from tensorflow_asr.utils import file_util

file_util.ENABLE_PATH_PREPROCESS = False

repodir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


text = "i'm good but it would have broken down after ten miles of that hard trail dawn came while they wound over the crest of the range and with the sun in their faces they took the downgrade it was well into the morning before nash reached logan"
# text = "a b"


def test_char():
    config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "configs", "librispeech", "characters", "char.yml.j2")
    config = file_util.load_yaml(config_path, repodir=repodir)
    decoder_config = DecoderConfig(config["decoder_config"])
    featurizer = CharTokenizer(decoder_config=decoder_config)
    print(featurizer.num_classes)
    print(text)
    indices = featurizer.tokenize(text)
    print(indices.numpy())
    batch_indices = tf.stack([indices, indices], axis=0)
    reversed_text = featurizer.detokenize(batch_indices)
    print(reversed_text.numpy())
    upoints = featurizer.detokenize_unicode_points(indices)
    print(upoints.numpy())


def test_wp():
    config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "configs", "librispeech", "wordpiece", "wp.yml.j2")
    config = file_util.load_yaml(config_path, repodir=repodir)
    decoder_config = DecoderConfig(config["decoder_config"])
    featurizer = WordPieceTokenizer(decoder_config=decoder_config)
    print(featurizer.num_classes)
    print(text)
    indices = featurizer.tokenize(text)
    print(indices.numpy())
    batch_indices = tf.stack([indices, indices], axis=0)
    reversed_text = featurizer.detokenize(batch_indices)
    print(reversed_text.numpy())
    upoints = featurizer.detokenize_unicode_points(indices)
    print(upoints.numpy())


def test_sp():
    config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "configs", "librispeech", "sentencepiece", "sp.yml.j2")
    config = file_util.load_yaml(config_path, repodir=repodir)
    decoder_config = DecoderConfig(config["decoder_config"])
    featurizer = SentencePieceTokenizer(decoder_config=decoder_config)
    print(featurizer.num_classes)
    print(text)
    indices = featurizer.tokenize(text)
    print(indices)
    indices = list(indices.numpy())
    indices += [0, 0]
    batch_indices = tf.stack([indices, indices], axis=0)
    reversed_text = featurizer.detokenize(batch_indices)
    print(reversed_text.numpy())
    upoints = featurizer.detokenize_unicode_points(indices)
    print(upoints.numpy())
