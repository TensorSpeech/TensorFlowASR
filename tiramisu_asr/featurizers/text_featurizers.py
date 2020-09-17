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

import abc
import codecs
import unicodedata
import tensorflow as tf
import tensorflow_datasets as tds

from ..utils.utils import preprocess_paths
from . import ENGLISH


class TextFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self, decoder_config: dict):
        self.scorer = None
        self.decoder_config = decoder_config
        if not self.decoder_config.get("vocabulary", None):
            self.decoder_config["vocabulary"] = ENGLISH  # Default language is english
        self.decoder_config["vocabulary"] = preprocess_paths(self.decoder_config["vocabulary"])
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None

    def _preprocess_text(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        return text.strip("\n")  # remove trailing newline

    def add_scorer(self, scorer: any = None):
        """ Add scorer to this instance """
        self.scorer = scorer

    def normalize_indices(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Remove -1 in indices by replacing them with blanks
        Args:
            indices (tf.Tensor): shape any

        Returns:
            tf.Tensor: normalized indices with shape same as indices
        """
        with tf.name_scope("normalize_indices"):
            minus_one = -1 * tf.ones_like(indices, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
            return tf.where(indices == minus_one, blank_like, indices)

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """ Prepand blank index for transducer models """
        return tf.concat([[self.blank], text], axis=0)

    @abc.abstractclassmethod
    def extract(self, text):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def iextract(self, indices):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def indices2upoints(self, indices):
        raise NotImplementedError()


class CharFeaturizer(TextFeaturizer):
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, decoder_config: dict):
        """
        decoder_config = {
            "vocabulary": str,
            "blank_at_zero": bool,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        super(CharFeaturizer, self).__init__(decoder_config)
        self._preprocess_vocabulary()

    def _preprocess_vocabulary(self):
        lines = []
        with codecs.open(self.decoder_config["vocabulary"], "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        self.blank = 0 if self.decoder_config["blank_at_zero"] else None
        self.tokens2indices = {}
        self.tokens = []
        index = 1 if self.blank == 0 else 0
        for line in lines:
            line = self._preprocess_text(line)
            if line.startswith("#") or not line: continue
            self.tokens2indices[line[0]] = index
            self.tokens.append(line[0])
            index += 1
        if self.blank is None: self.blank = len(self.tokens)  # blank not at zero
        self.vocab_array = self.tokens.copy()
        self.tokens.insert(self.blank, "")  # add blank token to tokens
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)
        self.upoints = tf.squeeze(
            tf.strings.unicode_decode(
                self.tokens, "UTF-8").to_tensor(shape=[None, 1])
        )

    def extract(self, text: str) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self._preprocess_text(text)
        text = list(text.strip())  # remove trailing space
        indices = [self.tokens2indices[token] for token in text]
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        tokens = tf.gather_nd(self.tokens, tf.expand_dims(indices, axis=-1))
        tokens = tf.strings.reduce_join(tokens, axis=-1)
        return tokens

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], dtype=tf.int32)
        ]
    )
    def indices2upoints(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return upoints


class SubwordFeaturizer(TextFeaturizer):
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, decoder_config: dict, subwords=None):
        """
        decoder_config = {
            "vocabulary": str,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        super(SubwordFeaturizer, self).__init__(decoder_config)
        self.subwords = subwords
        self.blank = 0  # subword treats blank as 0
        self.num_classes = self.subwords.vocab_size

    @classmethod
    def build_from_corpus(cls, decoder_config: dict, corpus_files: list):
        def corpus_generator():
            for file in corpus_files:
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    lines = lines[1:]
                for line in lines:
                    line = line.split("\t")
                    yield line[-1]
        subwords = tds.features.text.SubwordTextEncoder.build_from_corpus(
            corpus_generator(),
            decoder_config.get("target_vocab_size", 4096),
            decoder_config.get("max_subword_length", 4),
            decoder_config.get("max_corpus_chars", None),
            decoder_config.get("reserved_tokens", None)
        )
        return cls(decoder_config, subwords)

    @classmethod
    def load_from_file(cls, decoder_config: dict, filename_prefix: str):
        subwords = tds.features.text.SubwordTextEncoder.load_from_file(filename_prefix)
        return cls(decoder_config, subwords)

    def extract(self, text: str) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self._preprocess_text(text)
        text = text.strip()  # remove trailing space
        indices = self.subwords.encode(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        text = self.subwords.decode(indices)
        return tf.convert_to_tensor(text, dtype=tf.string)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], dtype=tf.int32)
        ]
    )
    def indices2upoints(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        pass
