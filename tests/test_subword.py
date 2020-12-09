import argparse
import tensorflow as tf

from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer

parser = argparse.ArgumentParser(prog="test subword")

parser.add_argument("transcripts", nargs="+", type=str, default=[None])

args = parser.parse_args()

config = {
    "vocabulary": None,
    "target_vocab_size": 1024,
    "max_subword_length": 4,
    "blank_at_zero": True,
    "beam_width": 5,
    "norm_score": True
}

text_featurizer = SubwordFeaturizer.build_from_corpus(config, args.transcripts)

print(len(text_featurizer.subwords.subwords))
print(text_featurizer.upoints)
print(text_featurizer.num_classes)

a = text_featurizer.extract("hello world")

print(a)

b = text_featurizer.indices2upoints(a)

tf.print(tf.strings.unicode_encode(b, "UTF-8"))
