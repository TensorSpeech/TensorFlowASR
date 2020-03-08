from __future__ import absolute_import

import codecs
import tensorflow as tf


class TextFeaturizer:
  """ Extract text feature based on char-level granularity.
  By looking up the vocabulary table, each line of transcript will be
  converted to a sequence of integer indexes.
  """

  def __init__(self, vocab_file):
    self.num_classes = 0
    lines = []
    with codecs.open(vocab_file, "r", "utf-8") as fin:
      lines.extend(fin.readlines())
    self.token_to_index = {}
    self.index_to_token = {}
    index = 0  # blank index = -1
    for line in lines:
      line = line[:-1]  # Strip the '\n' char
      # Skip comment line, empty line
      if line.startswith("#") or not line or line == "\n":
        continue
      self.token_to_index[line[0]] = index
      self.index_to_token[index] = line[0]
      index += 1
    self.num_classes = index + 1  # blank index is added later

  def compute_label_features(self, text):
    # Convert string to a list of integers
    tokens = list(text.strip().lower())
    feats = [self.token_to_index[token] for token in tokens]
    return tf.convert_to_tensor(feats, dtype=tf.int32)

# class UnicodeFeaturizer:
#     def __init__(self)
