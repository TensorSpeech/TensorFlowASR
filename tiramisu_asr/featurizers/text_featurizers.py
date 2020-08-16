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

import codecs
import unicodedata
import tensorflow as tf

from ..utils.utils import preprocess_paths
from . import ENGLISH


class TextFeaturizer:
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
        self.decoder_config = decoder_config

        if not self.decoder_config["vocabulary"]:
            self.decoder_config["vocabulary"] = ENGLISH  # Default language is english
        self.decoder_config["vocabulary"] = preprocess_paths(self.decoder_config["vocabulary"])

        self.scorer = None

        self.num_classes = 0
        lines = []
        with codecs.open(self.decoder_config["vocabulary"], "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_array = []
        self.tf_vocab_array = tf.constant([], dtype=tf.string)
        self.index_to_unicode_points = tf.constant([], dtype=tf.int32)
        index = 0
        if self.decoder_config["blank_at_zero"]:
            self.blank = 0
            index = 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [""]], axis=0)
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, [0]], axis=0)
        for line in lines:
            line = line[:-1]  # Strip the '\n' char
            # Skip comment line, empty line
            if line.startswith("#") or not line or line == "\n":
                continue
            self.token_to_index[line[0]] = index
            self.index_to_token[index] = line[0]
            self.vocab_array.append(line[0])
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [line[0]]], axis=0)
            upoint = tf.strings.unicode_decode(line[0], "UTF-8")
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, upoint], axis=0)
            index += 1
        self.num_classes = index
        if not self.decoder_config["blank_at_zero"]:
            self.blank = index
            self.num_classes += 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [""]], axis=0)
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, [0]], axis=0)

    def add_scorer(self, scorer: any = None):
        """ Add scorer to this instance, scorer can use decoder_config property """
        self.scorer = scorer

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """ Prepand blank index for transducer models """
        return tf.concat([[self.blank], text], axis=0)

    def extract(self, text: str) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
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
        """
        Convert list of integers to string
        Args:
            feat: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        minus_one = -1 * tf.ones_like(feat, dtype=tf.int32)
        blank_like = self.blank * tf.ones_like(feat, dtype=tf.int32)
        feat = tf.where(feat == minus_one, blank_like, feat)
        return tf.map_fn(self._idx_to_char, feat,
                         fn_output_signature=tf.TensorSpec([], tf.string))

    def _idx_to_char(self, arr: tf.Tensor) -> tf.Tensor:
        transcript = tf.constant("", dtype=tf.string)
        for i in arr:
            transcript = tf.strings.join([transcript, self.tf_vocab_array[i]])
        return transcript

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], dtype=tf.int32)
        ]
    )
    def index2upoints(self, feat: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        TFLite Map_fn Issue: https://github.com/tensorflow/tensorflow/issues/40221
        Only use in tf-nightly
        Args:
            feat: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        # filter -1 value to avoid outofrange
        minus_one = -1 * tf.ones_like(feat, dtype=tf.int32)
        blank_like = self.blank * tf.ones_like(feat, dtype=tf.int32)
        feat = tf.where(feat == minus_one, blank_like, feat)
        return tf.map_fn(lambda i: self.index_to_unicode_points[i], feat,
                         fn_output_signature=tf.TensorSpec([], dtype=tf.int32))
