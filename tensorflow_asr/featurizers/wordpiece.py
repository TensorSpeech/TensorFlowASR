# coding=utf-8
# Copyright 2021 TF.Text Authors.
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

"""Algorithm for learning wordpiece vocabulary."""

import re
import collections
from typing import List, Optional

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor

import numpy as np
import tensorflow_text as tft

Params = collections.namedtuple("Params", [
    "upper_thresh", "lower_thresh", "num_iterations", "max_input_tokens",
    "max_token_length", "max_unique_chars", "vocab_size", "slack_ratio",
    "include_joiner_token", "joiner", "reserved_tokens"
])


def extract_char_tokens(word_counts):
    """Extracts all single-character tokens from word_counts.
    Args:
      word_counts: list of (string, int) tuples
    Returns:
      set of single-character strings contained within word_counts
    """

    seen_chars = set()
    for word, _ in word_counts:
        for char in word:
            seen_chars.add(char)
    return seen_chars


def ensure_all_tokens_exist(input_tokens, output_tokens, include_joiner_token,
                            joiner):
    """Adds all tokens in input_tokens to output_tokens if not already present.
    Args:
      input_tokens: set of strings (tokens) we want to include
      output_tokens: string to int dictionary mapping token to count
      include_joiner_token: bool whether to include joiner token
      joiner: string used to indicate suffixes
    Returns:
      string to int dictionary with all tokens in input_tokens included
    """

    for token in input_tokens:
        if token not in output_tokens:
            output_tokens[token] = 1

        if include_joiner_token:
            joined_token = joiner + token
            if joined_token not in output_tokens:
                output_tokens[joined_token] = 1

    return output_tokens


def get_split_indices(word, curr_tokens, include_joiner_token, joiner):
    """Gets indices for valid substrings of word, for iterations > 0.
    For iterations > 0, rather than considering every possible substring, we only
    want to consider starting points corresponding to the start of wordpieces in
    the current vocabulary.
    Args:
      word: string we want to split into substrings
      curr_tokens: string to int dict of tokens in vocab (from previous iteration)
      include_joiner_token: bool whether to include joiner token
      joiner: string used to indicate suffixes
    Returns:
      list of ints containing valid starting indices for word
    """

    indices = []
    start = 0
    while start < len(word):
        end = len(word)
        while end > start:
            subtoken = word[start:end]
            # Subtoken includes the joiner token.
            if include_joiner_token and start > 0:
                subtoken = joiner + subtoken
            # If subtoken is part of vocab, "end" is a valid start index.
            if subtoken in curr_tokens:
                indices.append(end)
                break
            end -= 1

        if end == start:
            return None
        start = end

    return indices


def get_search_threshs(word_counts, upper_thresh, lower_thresh):
    """Clips the thresholds for binary search based on current word counts.
    The upper threshold parameter typically has a large default value that can
    result in many iterations of unnecessary search. Thus we clip the upper and
    lower bounds of search to the maximum and the minimum wordcount values.
    Args:
      word_counts: list of (string, int) tuples
      upper_thresh: int, upper threshold for binary search
      lower_thresh: int, lower threshold for binary search
    Returns:
      upper_search: int, clipped upper threshold for binary search
      lower_search: int, clipped lower threshold for binary search
    """

    counts = [count for _, count in word_counts]
    max_count = max(counts)
    min_count = min(counts)

    if upper_thresh is None:
        upper_search = max_count
    else:
        upper_search = max_count if max_count < upper_thresh else upper_thresh

    if lower_thresh is None:
        lower_search = min_count
    else:
        lower_search = min_count if min_count > lower_thresh else lower_thresh

    return upper_search, lower_search


def get_input_words(word_counts, reserved_tokens, max_token_length):
    """Filters out words that are longer than max_token_length or are reserved.
    Args:
      word_counts: list of (string, int) tuples
      reserved_tokens: list of strings
      max_token_length: int, maximum length of a token
    Returns:
      list of (string, int) tuples of filtered wordcounts
    """

    all_counts = []

    for word, count in word_counts:
        if len(word) > max_token_length or word in reserved_tokens:
            continue
        all_counts.append((word, count))

    return all_counts


def get_allowed_chars(all_counts, max_unique_chars):
    """Get the top max_unique_chars characters within our wordcounts.
    We want each character to be in the vocabulary so that we can keep splitting
    down to the character level if necessary. However, in order not to inflate
    our vocabulary with rare characters, we only keep the top max_unique_chars
    characters.
    Args:
      all_counts: list of (string, int) tuples
      max_unique_chars: int, maximum number of unique single-character tokens
    Returns:
      set of strings containing top max_unique_chars characters in all_counts
    """

    char_counts = collections.defaultdict(int)

    for word, count in all_counts:
        for char in word:
            char_counts[char] += count

    # Sort by count, then alphabetically.
    sorted_counts = sorted(sorted(char_counts.items(), key=lambda x: x[0]),
                           key=lambda x: x[1], reverse=True)

    allowed_chars = set()
    for i in range(min(len(sorted_counts), max_unique_chars)):
        allowed_chars.add(sorted_counts[i][0])
    return allowed_chars


def filter_input_words(all_counts, allowed_chars, max_input_tokens):
    """Filters out words with unallowed chars and limits words to max_input_tokens.
    Args:
      all_counts: list of (string, int) tuples
      allowed_chars: list of single-character strings
      max_input_tokens: int, maximum number of tokens accepted as input
    Returns:
      list of (string, int) tuples of filtered wordcounts
    """
    # Ensure that the input is sorted so that if `max_input_tokens` is reached
    # the least common tokens are dropped.
    all_counts = sorted(
        all_counts, key=lambda word_and_count: word_and_count[1], reverse=True)
    filtered_counts = []
    for word, count in all_counts:
        if (max_input_tokens != -1 and
                len(filtered_counts) >= max_input_tokens):
            break
        has_unallowed_chars = False
        for char in word:
            if char not in allowed_chars:
                has_unallowed_chars = True
                break
        if has_unallowed_chars:
            continue
        filtered_counts.append((word, count))

    return filtered_counts


def generate_final_vocabulary(reserved_tokens, char_tokens, curr_tokens):
    """Generates final vocab given reserved, single-character, and current tokens.
    Args:
      reserved_tokens: list of strings (tokens) that must be included in vocab
      char_tokens: set of single-character strings
      curr_tokens: string to int dict mapping token to count
    Returns:
      list of strings representing final vocabulary
    """

    sorted_char_tokens = sorted(list(char_tokens))
    vocab_char_arrays = []
    vocab_char_arrays.extend(reserved_tokens)
    vocab_char_arrays.extend(sorted_char_tokens)

    # Sort by count, then alphabetically.
    sorted_tokens = sorted(sorted(curr_tokens.items(), key=lambda x: x[0]),
                           key=lambda x: x[1], reverse=True)
    for token, _ in sorted_tokens:
        vocab_char_arrays.append(token)

    seen_tokens = set()
    # Adding unique tokens to list to maintain sorted order.
    vocab_words = []
    for word in vocab_char_arrays:
        if word in seen_tokens:
            continue
        seen_tokens.add(word)
        vocab_words.append(word)

    return vocab_words


def learn_with_thresh(word_counts, thresh, params):
    """Wordpiece learning algorithm to produce a vocab given frequency threshold.
    Args:
      word_counts: list of (string, int) tuples
      thresh: int, frequency threshold for a token to be included in the vocab
      params: Params namedtuple, parameters for learning
    Returns:
      list of strings, vocabulary generated for the given thresh
    """

    # Set of single-character tokens.
    char_tokens = extract_char_tokens(word_counts)
    curr_tokens = ensure_all_tokens_exist(char_tokens, {},
                                          params.include_joiner_token,
                                          params.joiner)

    for iteration in range(params.num_iterations):
        subtokens = [dict() for _ in range(params.max_token_length + 1)]
        # Populate array with counts of each subtoken.
        for word, count in word_counts:
            if iteration == 0:
                split_indices = range(1, len(word) + 1)
            else:
                split_indices = get_split_indices(word, curr_tokens,
                                                  params.include_joiner_token,
                                                  params.joiner)
                if not split_indices:
                    continue

            start = 0
            for index in split_indices:
                for end in range(start + 1, len(word) + 1):
                    subtoken = word[start:end]
                    length = len(subtoken)
                    if params.include_joiner_token and start > 0:
                        subtoken = params.joiner + subtoken
                    if subtoken in subtokens[length]:
                        # Subtoken exists, increment count.
                        subtokens[length][subtoken] += count
                    else:
                        # New subtoken, add to dict.
                        subtokens[length][subtoken] = count
                start = index

        next_tokens = {}
        # Get all tokens that have a count above the threshold.
        for length in range(params.max_token_length, 0, -1):
            for token, count in subtokens[length].items():
                if count >= thresh:
                    next_tokens[token] = count
                # Decrement the count of all prefixes.
                if len(token) > length:  # This token includes the joiner.
                    joiner_len = len(params.joiner)
                    for i in range(1 + joiner_len, length + joiner_len):
                        prefix = token[0:i]
                        if prefix in subtokens[i - joiner_len]:
                            subtokens[i - joiner_len][prefix] -= count
                else:
                    for i in range(1, length):
                        prefix = token[0:i]
                        if prefix in subtokens[i]:
                            subtokens[i][prefix] -= count

        # Add back single-character tokens.
        curr_tokens = ensure_all_tokens_exist(char_tokens, next_tokens,
                                              params.include_joiner_token,
                                              params.joiner)

    vocab_words = generate_final_vocabulary(params.reserved_tokens, char_tokens,
                                            curr_tokens)

    return vocab_words


def learn_binary_search(word_counts, lower, upper, params):
    """Performs binary search to find wordcount frequency threshold.
    Given upper and lower bounds and a list of (word, count) tuples, performs
    binary search to find the threshold closest to producing a vocabulary
    of size vocab_size.
    Args:
      word_counts: list of (string, int) tuples
      lower: int, lower bound for binary search
      upper: int, upper bound for binary search
      params: Params namedtuple, parameters for learning
    Returns:
      list of strings, vocab that is closest to target vocab_size
    """
    thresh = (upper + lower) // 2
    current_vocab = learn_with_thresh(word_counts, thresh, params)
    current_vocab_size = len(current_vocab)

    # Allow count to be within k% of the target count, where k is slack ratio.
    slack_count = params.slack_ratio * params.vocab_size
    if slack_count < 0:
        slack_count = 0

    is_within_slack = (current_vocab_size <= params.vocab_size) and (
        params.vocab_size - current_vocab_size <= slack_count)

    # We"ve created a vocab within our goal range (or, ran out of search space).
    if is_within_slack or lower >= upper or thresh <= 1:
        return current_vocab

    current_vocab = None

    if current_vocab_size > params.vocab_size:
        return learn_binary_search(word_counts, thresh + 1, upper, params)

    else:
        return learn_binary_search(word_counts, lower, thresh - 1, params)


def count_words(iterable) -> collections.Counter:
    """Converts a iterable of arrays of words into a `Counter` of word counts."""
    counts = collections.Counter()
    for words in iterable:
        # Convert a RaggedTensor to a flat/dense Tensor.
        words = getattr(words, "flat_values", words)
        # Flatten any dense tensor
        words = np.reshape(words, [-1])
        counts.update(words)

    # Decode the words if necessary.
    example_word = next(iter(counts.keys()))
    if isinstance(example_word, bytes):
        counts = collections.Counter(
            {word.decode("utf-8"): count for word, count in counts.items()})

    return counts


def learn(word_counts,
          vocab_size: int,
          reserved_tokens: List[str] = [],
          upper_thresh: Optional[int] = int(1e7),
          lower_thresh: Optional[int] = 10,
          num_iterations: int = 4,
          max_input_tokens: Optional[int] = int(5e6),
          max_token_length: int = 50,
          max_unique_chars: int = 1000,
          slack_ratio: float = 0.05,
          include_joiner_token: bool = True,
          joiner: str = "##") -> List[str]:
    """Takes in wordcounts and returns wordpiece vocabulary.
    Args:
      word_counts: (word, count) pairs as a dictionary, or list of tuples.
      vocab_size: The target vocabulary size. This is the maximum size.
      reserved_tokens: A list of tokens that must be included in the vocabulary.
      upper_thresh: Initial upper bound on the token frequency threshold.
      lower_thresh: Initial lower bound on the token frequency threchold.
      num_iterations: Number of iterations to run.
      max_input_tokens: The maximum number of words in the initial vocabulary. The
        words with the lowest counts are discarded. Use `None` or `-1` for "no
        maximum".
      max_token_length: The maximum token length. Counts for longer words are
        discarded.
      max_unique_chars: The maximum alphabet size. This prevents rare characters
        from inflating the vocabulary. Counts for words containing characters
        ouside of the selected alphabet are discarded.
      slack_ratio: The maximum deviation acceptable from `vocab_size` for an
        acceptable vocabulary. The acceptable range of vocabulary sizes is from
        `vocab_size*(1-slack_ratio)` to `vocab_size`.
      include_joiner_token: If true, include the `joiner` token in the output
        vocabulary.
      joiner: The prefix to include on suffix tokens in the output vocabulary.
        Usually "##". For example "places" could be tokenized as `["place",
        "##s"]`.
    Returns:
      string, final vocabulary with each word separated by newline
    """
    if isinstance(word_counts, dict):
        word_counts = word_counts.items()

    params = Params(upper_thresh, lower_thresh, num_iterations, max_input_tokens,
                    max_token_length, max_unique_chars, vocab_size, slack_ratio,
                    include_joiner_token, joiner, reserved_tokens)

    upper_search, lower_search = get_search_threshs(word_counts,
                                                    params.upper_thresh,
                                                    params.lower_thresh)
    all_counts = get_input_words(word_counts, params.reserved_tokens,
                                 params.max_token_length)
    allowed_chars = get_allowed_chars(all_counts, params.max_unique_chars)

    filtered_counts = filter_input_words(all_counts, allowed_chars,
                                         params.max_input_tokens)

    vocab = learn_binary_search(filtered_counts, lower_search, upper_search,
                                params)

    return vocab


def build_word_counts(corpus_generator):
    counts = {}
    for transcript in corpus_generator:
        words = transcript.split()
        for word in words:
            if counts.get(word, None) is None:
                counts[word] = 0
            else:
                counts[word] += 1
    return counts


def build_from_corpus(corpus_generator,
                      target_vocab_size: int,
                      output_file_path: str,
                      max_subword_length: int = 50,
                      max_corpus_chars: int = None,
                      reserved_tokens: List[str] = [],
                      num_iterations: int = 4):
    word_counts = build_word_counts(corpus_generator)
    max_corpus_chars = max_corpus_chars or 1e7
    reserved_tokens = reserved_tokens or []
    vocab = learn(word_counts, target_vocab_size,
                  reserved_tokens=reserved_tokens, num_iterations=num_iterations,
                  max_input_tokens=10000000, max_token_length=max_subword_length, max_unique_chars=max_corpus_chars)
    with open(output_file_path, "w") as f:
        for token in vocab: print(token, file=f)


class WordpieceTokenizer(tft.WordpieceTokenizer):
    @property
    def vocab_size(self):
        vocab, _ = self._get_vocab_and_ids()
        return tf.shape(vocab)[0].numpy()

    def _get_vocab_and_ids(self):
        export = getattr(self._vocab_lookup_table, 'export', None)
        if export is None:
            table = getattr(self._vocab_lookup_table, '_table')
            export = table.export

        vocab, ids = export()  # pylint: disable=protected-access

        # `.export` doesn't set the shapes.
        vocab = check_ops.ensure_shape(vocab, [
            None,
        ])
        ids = check_ops.ensure_shape(ids, [
            None,
        ])

        order = sort_ops.argsort(ids)

        ids = array_ops.gather(ids, order)
        vocab = array_ops.gather(vocab, order)

        return vocab, ids

    def detokenize(self, token_ids):
        r"""Convert a `Tensor` or `RaggedTensor` of wordpiece IDs to string-words.
        >>> import pathlib
        >>> pathlib.Path('vocab.txt').write_text(
        ...     "a b c ##a ##b ##c".replace(' ', '\n'))
        >>> wordpiece = text.WordpieceTokenizer('vocab.txt')
        >>> token_ids = [[0, 4, 5, 2, 5, 5, 5]]
        >>> wordpiece.detokenize(token_ids)
        <tf.RaggedTensor [[b'ab', b'cccc']]>
        The word pieces are joined along the innermost axis to make words. So the
        result has the same rank as the input, but the innermost axis of the result
        indexes words instead of word pieces.
        The shape transformation is: `[..., wordpieces] => [..., words]`
        When the input shape is `[..., words, wordpieces]` (like the output of
        `WordpieceTokenizer.tokenize`) the result's shape is `[..., words, 1]`.
        The additional ragged axis can be removed using `words.merge_dims(-2, -1)`.
        Note: This method assumes wordpiece IDs are dense on the interval
        `[0, vocab_size)`.
        Args:
          token_ids: A `RaggedTensor` or `Tensor` with an int dtype. Must have
          `ndims >= 2`
        Returns:
          A `RaggedTensor` with dtype `string` and the rank as the input
          `token_ids`.
        """
        # If there are performance issues with this method or problems with lookup
        # tables using sparse IDs see the notes in b/177610044.
        vocab, ids = self._get_vocab_and_ids()

        first_is_zero = tf.math.equal(ids[0], 0)
        steps = ids[1:] - ids[:-1]
        all_one_step = tf.reduce_all(tf.math.equal(steps, 1))

        check = control_flow_ops.Assert(
            first_is_zero & all_one_step,
            data=[('`detokenize` only works with vocabulary tables where the '
                   'indices are dense on the interval `[0, vocab_size)`')])
        with ops.control_dependencies([check]):
            token_ids = tf.math.minimum(
                token_ids,
                # Limit the OOV buckets to a single index.
                tf.cast(array_ops.size(vocab), token_ids.dtype))

        # Add the unknown token at that index.
        vocab = array_ops.concat([vocab, [self._unknown_token]], axis=0)

        # Lookup the text tokens and join them along the innermost axis.
        txt_tokens = array_ops.gather(vocab, token_ids)

        # Ensure the input is Ragged.
        if not isinstance(txt_tokens, RaggedTensor):
            txt_tokens = RaggedTensor.from_tensor(txt_tokens)

        # Join the tokens along the last axis.
        words = string_ops.reduce_join_v2(txt_tokens, axis=-1, separator=' ')

        # Collapse " ##" in all strings to make words.
        words = string_ops.regex_replace(
            words, ' ' + re.escape(self._suffix_indicator), '')

        # Strip leading and trailing spaces.
        words = string_ops.regex_replace(words, '^ +| +$', '')

        # Split on spaces so the last axis is "words".
        words = ragged_string_ops.string_split_v2(words, sep=' ')
        return words
