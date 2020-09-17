import argparse
import tensorflow as tf
import tensorflow_datasets as tds

parser = argparse.ArgumentParser(prog="test subword")

parser.add_argument("transcripts", nargs="+", type=str, default=[None])

args = parser.parse_args()


def corpus_generator():
    for file in args.transcripts:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            lines = lines[1:]
        for line in lines:
            line = line.split("\t")
            yield line[-1]


subwords = tds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_generator(), target_vocab_size=1024, max_subword_length=4,
)

print(subwords.vocab_size)

print(subwords.subwords)

a = subwords.encode("hello world")

decode = tf.function(subwords.decode)

print(decode(a))

print(decode([0, 0, 0, 0]))

subwords.save_to_file("/tmp/subwords.txt")
