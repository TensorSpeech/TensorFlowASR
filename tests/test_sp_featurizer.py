# pylint: disable=line-too-long
import os

import tensorflow as tf

from tensorflow_asr.configs.config import DecoderConfig
from tensorflow_asr.featurizers.text_featurizers import SentencePieceFeaturizer
from tensorflow_asr.utils import file_util

file_util.ENABLE_PATH_PREPROCESS = False

config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "configs", "sp.yml.j2")
print(config_path)
config = file_util.load_yaml(config_path)

decoder_config = DecoderConfig(config["decoder_config"])

text = "i'm good but it would have broken down after ten miles of that hard trail dawn came while they wound over the crest of the range and with the sun in their faces they took the downgrade it was well into the morning before nash reached logan"
# text = "a b"


def test():
    featurizer = SentencePieceFeaturizer(decoder_config=decoder_config)
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
