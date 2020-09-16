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

import tensorflow as tf
import tensorflow_datasets as tds

from .text_featurizers import TextFeaturizer


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
            "blank_at_zero": bool,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        TextFeaturizer.__init__(decoder_config)
        self.subwords = subwords
        self.blank = 0
        self.num_classes = self.subwords.vocab_size

    @classmethod
    def build_from_corpus(cls, decoder_config, corpus_generator, target_vocab_size,
                          max_subword_length=20, max_corpus_chars=None, reserved_tokens=None):
        subwords = tds.features.text.SubwordTextEncoder.build_from_corpus(
            corpus_generator, target_vocab_size, max_subword_length,
            max_corpus_chars, reserved_tokens
        )
        return cls(decoder_config, subwords)

    @classmethod
    def load_from_file(cls, decoder_config, filename_prefix):
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
        text = list(text.strip())  # remove trailing space
        return self.subwords.encode(text)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        return self.subwords.decode(indices)

    def indices2tokens(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert indices to text
        Args:
            indices (tf.Tensor): shape [B, None]

        Returns:
            tf.Tensor: text with shape [B] with dtype string
        """
        with tf.name_scope("indices2tokens"):
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
