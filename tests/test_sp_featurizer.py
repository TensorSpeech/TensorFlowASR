# pylint: disable=line-too-long
import os

import tensorflow as tf

from tensorflow_asr.configs.config import DecoderConfig
from tensorflow_asr.featurizers.text_featurizers import SentencePieceFeaturizer

decoder_config = DecoderConfig(
    {
        "model_type": "unigram",
        "vocabulary": f"{os.path.dirname(__file__)}/../vocabularies/librispeech/sentencepiece/train_uni_1000.model",
        "blank_index": 0,
        "pad_token": "<pad>",
        "pad_index": 0,
        "unknown_token": "<unk>",
        "unknown_index": 1,
        "bos_token": "<s>",
        "bos_index": 2,
        "eos_token": "</s>",
        "eos_index": 3,
    }
)

text = "i'm good but it would have broken down after ten miles of that hard trail dawn came while they wound over the crest of the range and with the sun in their faces they took the downgrade it was well into the morning before nash reached logan"


def test():
    featurizer = SentencePieceFeaturizer(decoder_config=decoder_config)
    print(text)
    indices = featurizer.extract(text)
    print(indices.numpy())
    indices = featurizer.tf_extract(text)
    print(indices.numpy())
    batch_indices = tf.stack([indices, indices], axis=0)
    reversed_text = featurizer.iextract(batch_indices)
    print(reversed_text.numpy())
    upoints = featurizer.indices2upoints(indices)
    print(upoints.numpy())
