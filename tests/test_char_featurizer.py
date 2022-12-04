# pylint: disable=line-too-long
import os

import tensorflow as tf

from tensorflow_asr.configs.config import DecoderConfig
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer

decoder_config = DecoderConfig(
    {
        "vocabulary": f"{os.path.dirname(__file__)}/../vocabularies/english.characters",
    }
)

text = "i'm good but it would have broken down after ten miles of that hard trail dawn came while they wound over the crest of the range and with the sun in their faces they took the downgrade it was well into the morning before nash reached logan"


def test():
    featurizer = CharFeaturizer(decoder_config=decoder_config)
    print(featurizer.tokens)
    print(featurizer.num_classes)
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
