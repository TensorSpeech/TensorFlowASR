# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import codecs
import unicodedata
import numpy as np
import multiprocessing
import tensorflow as tf

from ..utils.utils import preprocess_paths
from ..datasets import base_dataset

DEFAULT_VOCAB = os.path.abspath(os.path.join(os.path.dirname(base_dataset.__file__), "vocabulary.txt"))


class TextFeaturizer:
    """ Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, vocab_file):
        if not vocab_file: vocab_file = DEFAULT_VOCAB
        self.num_classes = 0
        lines = []
        with codecs.open(preprocess_paths(vocab_file), "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_array = []
        index = 0  # blank index = -1
        for line in lines:
            line = line[:-1]  # Strip the '\n' char
            # Skip comment line, empty line
            if line.startswith("#") or not line or line == "\n":
                continue
            self.token_to_index[line[0]] = index
            self.index_to_token[index] = line[0]
            self.vocab_array.append(line[0])
            index += 1
        self.num_classes = index + 1  # blank index is added later

    def extract(self, text: str) -> tf.Tensor:
        # Convert string to a list of integers
        text = unicodedata.normalize("NFC", text.lower())
        tokens = list(text.strip())
        new_tokens = []
        for tok in tokens:
            if tok in self.vocab_array:
                new_tokens.append(tok)
        tokens = new_tokens
        feats = [self.token_to_index[token] for token in tokens]
        return tf.convert_to_tensor(feats, dtype=tf.int32)

    def iextract(self, feat: tf.Tensor) -> tf.Tensor:
        feat = feat.numpy()

        with multiprocessing.Pool(len(feat)) as pool:
            results = pool.map(self._idx_to_char, feat)

        return tf.convert_to_tensor(np.concatenate(results), dtype=tf.string)

    def _idx_to_char(self, arr: np.ndarray) -> np.ndarray:
        arr = arr[arr != self.num_classes - 1]
        return np.array(["".join([self.index_to_token[i] for i in arr])])

# class UnicodeFeaturizer:
#     def __init__(self)
