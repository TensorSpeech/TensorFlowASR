# Copyright 2020 Huy Le Nguyen (@nglehuy)
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
import multiprocessing
import os
import unicodedata
from dataclasses import asdict, dataclass

import sentencepiece as sp
import tensorflow as tf
import tensorflow_text as tft
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from tensorflow_asr.configs import Config, DecoderConfig
from tensorflow_asr.utils import file_util

logger = tf.get_logger()


@dataclass
class TOKENIZER_TYPES:
    CHARACTERS: str = "characters"
    WORDPIECE: str = "wordpiece"
    SENTENCEPIECE: str = "sentencepiece"


ENGLISH_CHARACTERS = [
    "<blank>",
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
]


def get(config: Config):
    if config.decoder_config.type == TOKENIZER_TYPES.SENTENCEPIECE:
        logger.info("Loading SentencePiece model ...")
        tokenizer = SentencePieceTokenizer(config.decoder_config)
    elif config.decoder_config.type == TOKENIZER_TYPES.WORDPIECE:
        logger.info("Loading wordpiece ...")
        tokenizer = WordPieceTokenizer(config.decoder_config)
    elif config.decoder_config.type == TOKENIZER_TYPES.CHARACTERS:
        logger.info("Use characters ...")
        tokenizer = CharTokenizer(config.decoder_config)
    else:
        raise ValueError(f"type must be in {asdict(TOKENIZER_TYPES()).values()}, received {config.decoder_config.type}")
    return tokenizer


def build(config: Config):
    if config.decoder_config.type == TOKENIZER_TYPES.CHARACTERS:
        CharTokenizer.build_from_corpus(config.decoder_config)
        return
    if config.decoder_config.type == TOKENIZER_TYPES.SENTENCEPIECE:
        SentencePieceTokenizer.build_from_corpus(config.decoder_config)
        return
    if config.decoder_config.type == TOKENIZER_TYPES.WORDPIECE:
        WordPieceTokenizer.build_from_corpus(config.decoder_config)
        return


class Tokenizer:
    def __init__(self, decoder_config: DecoderConfig):
        self.scorer = None
        self.decoder_config = decoder_config
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None
        self.max_length = 0

    @classmethod
    def corpus_generator(cls, decoder_config: DecoderConfig):
        for file_path in file_util.preprocess_paths(decoder_config.train_files):
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                for line in temp_lines[1:]:  # Skip the header of tsv file
                    data = line.split("\t", 2)[-1]  # get only transcript
                    data = cls.normalize_text(data, decoder_config).numpy()
                    yield data

    @property
    def shape(self) -> list:
        return [self.max_length if self.max_length > 0 else None]

    @property
    def prepand_shape(self) -> list:
        return [self.max_length + 1 if self.max_length > 0 else None]

    def update_length(
        self,
        length: int,
    ):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    @classmethod
    def normalize_text(cls, text: tf.Tensor, decoder_config: DecoderConfig):
        text = tf.strings.regex_replace(text, b"\xe2\x81\x87".decode("utf-8"), "")
        text = tft.normalize_utf8(text, decoder_config.normalization_form)
        text = tf.strings.regex_replace(text, r"\p{Cc}|\p{Cf}", " ")
        text = tf.strings.regex_replace(text, decoder_config.unknown_token, "")
        text = tf.strings.regex_replace(text, decoder_config.pad_token, "")
        text = tf.strings.regex_replace(text, r" +", " ")
        text = tf.strings.lower(text, encoding="utf-8")
        text = tf.strings.strip(text)  # remove trailing whitespace
        return text

    def add_scorer(self, scorer: any = None):
        """Add scorer to this instance"""
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
            return tf.where(tf.equal(indices, minus_one), blank_like, indices)

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """Prepand blank index for transducer models"""
        return tf.concat([[self.blank], text], 0)

    def tokenize(self, text: str) -> tf.Tensor:
        raise NotImplementedError()

    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def detokenize_unicode_points(self, indices: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()


class CharTokenizer(Tokenizer):
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, decoder_config: DecoderConfig):
        super().__init__(decoder_config)
        lines = []
        if self.decoder_config.vocabulary is not None:
            with codecs.open(self.decoder_config.vocabulary, "r", "utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = ENGLISH_CHARACTERS
        self.blank = self.decoder_config.blank_index
        self.tokens = []
        for line in lines:
            line = unicodedata.normalize(self.decoder_config.normalization_form, line.lower()).strip("\n")
            if line.startswith("#") or not line:
                continue
            self.tokens.append(line)
        if self.blank is None:
            self.blank = len(self.tokens)  # blank not at zero
        self.num_classes = len(self.tokens)
        self.indices = tf.range(self.num_classes, dtype=tf.int32)
        self.tokenizer = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=self.tokens, values=self.indices, key_dtype=tf.string, value_dtype=tf.int32),
            default_value=self.blank,
        )
        self.detokenizer = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=self.indices, values=self.tokens, key_dtype=tf.int32, value_dtype=tf.string),
            default_value=self.tokens[self.blank],
        )
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(shape=[None, 1])

    @classmethod
    def build_from_corpus(cls, decoder_config: DecoderConfig):
        if os.path.exists(decoder_config.vocabulary):
            return cls(decoder_config)

        def write_vocab_file(filepath, vocab):
            with tf.io.gfile.GFile(filepath, "w") as f:
                for token in vocab:
                    print(token, file=f)

        vocab = set()
        for text in cls.corpus_generator(decoder_config):
            vocab.update(text)

        write_vocab_file(decoder_config.vocabulary, vocab)

        return cls(decoder_config)

    def tokenize(self, text):
        text = self.normalize_text(text, self.decoder_config)
        text = tf.strings.unicode_split(text, "UTF-8")
        return self.tokenizer.lookup(text)

    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.blank))
        tokens = self.detokenizer.lookup(indices)
        tokens = tf.strings.reduce_join(tokens, axis=-1)
        tokens = self.normalize_text(tokens, self.decoder_config)
        return tokens

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def detokenize_unicode_points(self, indices: tf.Tensor) -> tf.Tensor:
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
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, decoder_config: DecoderConfig):
        super().__init__(decoder_config)
        self.blank = self.decoder_config.blank_index
        self.tokenizer = tft.FastSentencepieceTokenizer(self.__load_model(), reverse=False, add_bos=False, add_eos=False)
        self.num_classes = int(self.tokenizer.vocab_size())

    def __load_model(self):
        with file_util.read_file(self.decoder_config.vocabulary) as path:
            with open(path, "rb") as f:
                return f.read()

    @classmethod
    def build_from_corpus(cls, decoder_config: DecoderConfig):
        if os.path.exists(decoder_config.vocabulary):
            return cls(decoder_config)

        sp.SentencePieceTrainer.Train(
            sentence_iterator=cls.corpus_generator(decoder_config),
            model_prefix=os.path.splitext(decoder_config.vocabulary)[0],
            model_type=decoder_config.model_type,
            vocab_size=decoder_config.vocab_size,
            hard_vocab_limit=True,
            unk_id=decoder_config.unknown_index,
            bos_id=decoder_config.bos_index,
            eos_id=decoder_config.eos_index,
            pad_id=decoder_config.pad_index,
            character_coverage=decoder_config.character_coverage,
            unk_surface="",  # change default unk surface U+2047("⁇") by ""
            allow_whitespace_only_pieces=False,
            split_by_whitespace=(not decoder_config.keep_whitespace),
            treat_whitespace_as_suffix=False,
            user_defined_symbols="",
            max_sentencepiece_length=decoder_config.max_sentencepiece_length,
            max_sentence_length=decoder_config.max_sentence_length,  # bytes
            remove_extra_whitespaces=True,
            num_threads=multiprocessing.cpu_count(),
        )

        return cls(decoder_config)

    def tokenize(self, text: tf.Tensor) -> tf.Tensor:
        text = self.normalize_text(text, self.decoder_config)
        indices = self.tokenizer.tokenize(text)
        indices = tf.cast(indices, tf.int32)
        return indices

    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.blank))
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.decoder_config.unknown_index))
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.decoder_config.bos_index))
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.decoder_config.eos_index))
        transcripts = self.tokenizer.detokenize(indices)
        transcripts = self.normalize_text(transcripts, self.decoder_config)
        # transcripts = tf.strings.regex_replace(transcripts, r" +", " ")
        return transcripts

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def detokenize_unicode_points(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            transcripts = self.detokenize(tf.reshape(indices, [1, -1]))
            upoints = tf.strings.unicode_decode(transcripts, "UTF-8").to_tensor()
            return tf.reshape(upoints, [-1])


class WordPieceTokenizer(Tokenizer):
    def __init__(self, decoder_config: DecoderConfig):
        super().__init__(decoder_config)
        self.blank = self.decoder_config.blank_index  # treat [PAD] as blank
        self.vocab = None
        with tf.io.gfile.GFile(self.decoder_config.vocabulary, "r") as voc:
            self.vocab = voc.read().splitlines()
        if not self.vocab:
            raise ValueError("Unable to read vocabulary")
        self.tokenizer = tft.FastWordpieceTokenizer(
            vocab=self.vocab,
            token_out_type=tf.int32,
            unknown_token=self.decoder_config.unknown_token,
            no_pretokenization=True,  # False is limited, so we manually do pretokenization
            support_detokenization=True,
        )
        self.num_classes = len(self.vocab)

    @classmethod
    def build_from_corpus(cls, decoder_config: DecoderConfig):
        if os.path.exists(decoder_config.vocabulary):
            return cls(decoder_config)

        def generator():
            for data in cls.corpus_generator(decoder_config):
                yield data

        def write_vocab_file(filepath, vocab):
            with tf.io.gfile.GFile(filepath, "w") as f:
                for token in vocab:
                    print(token, file=f)

        dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.string)).batch(1000).prefetch(2)
        vocab = bert_vocab.bert_vocab_from_dataset(
            dataset,
            vocab_size=decoder_config.vocab_size,
            reserved_tokens=decoder_config.reserved_tokens,
            bert_tokenizer_params={
                "lower_case": False,  # keep original from dataset
                "keep_whitespace": decoder_config.keep_whitespace,
                "normalization_form": decoder_config.normalization_form,
                "preserve_unused_token": False,
            },
            learn_params={
                "max_token_length": decoder_config.max_token_length,
                "max_unique_chars": decoder_config.max_unique_chars,
                "num_iterations": decoder_config.num_iterations,
            },
        )
        write_vocab_file(decoder_config.vocabulary, vocab)

        return cls(decoder_config)

    def tokenize(self, text: tf.Tensor) -> tf.Tensor:
        text = self.normalize_text(text, self.decoder_config)
        if self.decoder_config.keep_whitespace:
            text = tf.strings.regex_replace(text, " ", "| |")
            text = tf.strings.split(text, sep="|")
        else:
            text = tf.strings.split(text)
        indices = self.tokenizer.tokenize(text).merge_dims(0, 1)
        return indices

    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.blank))
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.decoder_config.unknown_index))
        transcripts = self.tokenizer.detokenize(indices)
        transcripts = self.normalize_text(transcripts, self.decoder_config)
        # transcripts = tf.strings.regex_replace(transcripts, r" +", " ")
        return transcripts

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def detokenize_unicode_points(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            transcripts = self.detokenize(tf.reshape(indices, [1, -1]))
            upoints = tf.strings.unicode_decode(transcripts, "UTF-8").to_tensor()
            return tf.reshape(upoints, [-1])
