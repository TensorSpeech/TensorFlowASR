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
"""Bigram language model standing in for a transducer's internal LM, for LODR beam search"""

import logging

import numpy as np

from tensorflow_asr import keras, tf
from tensorflow_asr.models.lm.language_model import LanguageModel

logger = logging.getLogger(__name__)


def count_bigrams(token_sequences, vocab_size: int, blank: int) -> np.ndarray:
    """
    Count adjacent token pairs over tokenized transcripts.

    `counts[v, w]` is how often `w` followed `v`. Row `blank` counts sentence starts: the beam
    search holds `last_token = blank` until a hypothesis emits its first label, so that row is
    what conditions the first token of an utterance. There is no end-of-sentence counterpart --
    a hypothesis ends when the encoder frames run out, never on an emitted symbol.

    Parameters
    ----------
    token_sequences : Iterable[Sequence[int]]
        Tokenized transcripts, as produced by `Tokenizer.tokenize`. Consumed once, so a generator
        is fine and is what the CLI passes.
    vocab_size : int
    blank : int

    Returns
    -------
    np.ndarray, shape [V, V], dtype int64
    """
    counts = np.zeros([vocab_size, vocab_size], dtype=np.int64)
    for tokens in token_sequences:
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        tokens = tokens[(tokens >= 0) & (tokens < vocab_size)]  # drop anything outside the vocabulary
        if tokens.size == 0:
            continue
        contexts = np.concatenate([[blank], tokens[:-1]])
        # `np.add.at` rather than `counts[contexts, tokens] += 1`: fancy-index assignment applies a
        # repeated pair only once, which would undercount every doubled token.
        np.add.at(counts, (contexts, tokens), 1)
    return counts


def build_table(counts: np.ndarray, blank: int, interpolation: float = 0.75, delta: float = 0.1) -> np.ndarray:
    """
    Turn bigram counts into the log-probability table `BigramLanguageModel` scores from.

    Smoothed by Jelinek-Mercer interpolation with an add-`delta` unigram:

        p_uni(w)   = (c(w) + delta) / sum_w' (c(w') + delta)
        p_bi(w|v)  = c(v, w) / c(v)
        p(w|v)     = lam * p_bi(w|v) + (1 - lam) * p_uni(w)

    Smoothing is not optional. An unsmoothed table gives `ln 0 = -inf` for every pair that never
    occurred, and the beam adds those scores into hypotheses that are perfectly legal, killing them
    outright. `delta > 0` guarantees every label keeps some mass.

    `lam` is `interpolation` for a context that was seen and **0 for one that was not**, so an
    unseen context falls back entirely to the unigram. Without that special case its row would sum
    to `1 - interpolation` instead of 1, and since a hypothesis moves between contexts as it grows,
    that missing mass would shift scores between competing hypotheses rather than cancelling.

    `interpolation` must stay strictly below 1. At exactly 1 a *seen* context keeps no unigram
    mass, so any label that never followed it lands on `ln 0 = -inf` -- the unseen-context guard
    above does not help, because that row does have counts. There is no safe "pure bigram" setting.

    Blank is excluded throughout: it is dropped from the successor counts, the labels are
    normalised to sum to one on their own, and the blank column of the result is exactly 0.0. That
    matches `_internal_lm_log_probs` in the transducer, so `lm_beta` means the same thing whether
    you subtract this table (LODR) or the transducer's own estimate (ILME).

    Parameters
    ----------
    counts : np.ndarray, shape [V, V]
        As returned by `count_bigrams`.
    blank : int
    interpolation : float
        `lam` above, in [0, 1). Higher trusts the counts more; 0 ignores them entirely and leaves a
        unigram model. 1 is rejected, see above.
    delta : float
        Add-`delta` mass on the unigram. Must be > 0.

    Returns
    -------
    np.ndarray, shape [V, V], dtype float32
        `table[v, w] = ln p(w | v)`, with `table[:, blank] = 0.0`.
    """
    if not 0.0 <= interpolation < 1.0:
        raise ValueError(
            f"interpolation must be in [0, 1), got {interpolation}. At 1 a seen context keeps no "
            "unigram mass, so every label it never saw would score ln 0 = -inf."
        )
    if delta <= 0:
        raise ValueError(f"delta must be > 0, otherwise an unseen label gets ln 0 = -inf, got {delta}")

    counts = np.asarray(counts, dtype=np.float64).copy()
    vocab_size = counts.shape[0]
    if counts.ndim != 2 or counts.shape[1] != vocab_size:
        raise ValueError(f"counts must be square [V, V], got {counts.shape}")
    counts[:, blank] = 0.0  # blank is never a successor

    successor = counts.sum(axis=0) + delta  # [V]
    successor[blank] = 0.0
    p_unigram = successor / successor.sum()  # [V], sums to 1 over labels

    row_total = counts.sum(axis=1, keepdims=True)  # [V, 1]
    p_bigram = counts / np.maximum(row_total, 1.0)
    lam = np.where(row_total > 0, interpolation, 0.0)  # [V, 1]
    probs = lam * p_bigram + (1.0 - lam) * p_unigram[None, :]

    table = np.zeros_like(probs)
    labels = np.arange(vocab_size) != blank
    table[:, labels] = np.log(probs[:, labels])
    return table.astype(np.float32)


@keras.utils.register_keras_serializable(package=__name__)
class BigramLanguageModel(LanguageModel):
    """
    Bigram LM over the transducer's own vocabulary, fitted by counting.

    This is the low-order LM that LODR (https://arxiv.org/abs/2203.16776) subtracts in place of the
    transducer's internal LM. Fit it on the **ASR training transcripts** -- the same text the
    transducer saw -- with `tensorflow_asr train_lm --target=internal`. Fitting it on the
    external LM's target-domain corpus instead would subtract the very thing fusion is adding.

    A bigram is not trained by gradient descent. Its maximum-likelihood estimate is a ratio of
    counts, available exactly and in one pass, so `fit_counts` replaces the usual `fit` and
    `train_lm.py` dispatches to it. The table is held as a **non-trainable weight**, which is what
    makes it save and load through the ordinary keras h5 path like any other model.

    A bigram also needs no recurrent state: it conditions only on the previous token, which the
    beam search already hands to `call_next`. `get_initial_state` returns a placeholder, kept only
    because the beam threads one state tensor per LM through its `tf.while_loop`. A higher-order
    n-gram would put the tokens before last in that tensor and otherwise work unchanged.

    Storage is a dense `[V, V]` float32 table -- 256 KiB at V = 256, 4 MiB at V = 1000. The
    20k-bigram pruning of the paper is not implemented: it exists for vocabularies far larger than
    this repository's, where dense storage stops being free.
    """

    def __init__(self, vocab_size: int, blank: int = 0, interpolation: float = 0.75, delta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.blank = blank
        self.interpolation = interpolation
        self.delta = delta
        # Created here rather than in `build` because the decoder calls this model from inside a
        # `tf.while_loop`, where creating variables fails. Non-trainable: counting sets it, and no
        # gradient should ever move it.
        self.table = self.add_weight(
            name="table",
            shape=[vocab_size, vocab_size],
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        # Every weight this model will ever have now exists, so it really is built. Saying so lets
        # `save_weights` / `load_weights` work without a dummy forward pass first, which keras
        # otherwise insists on.
        self.built = True

    def fit_counts(self, token_sequences):
        """
        Fit by counting, the closed-form maximum likelihood estimate. Returns the raw counts.

        `train_lm.py` calls this instead of `fit` when the model provides it, which is how the same
        script trains both a neural LM by gradient descent and this one by a single pass.
        """
        counts = count_bigrams(token_sequences, vocab_size=self.vocab_size, blank=self.blank)
        self.table.assign(build_table(counts, blank=self.blank, interpolation=self.interpolation, delta=self.delta))
        return counts

    def get_initial_state(self, batch_size):
        return tf.zeros([batch_size, 1], dtype=tf.int32)

    def call(self, tokens, training=False):
        return tf.gather(self.table, tokens)  # [B, U] => [B, U, V]

    def call_next(self, previous_tokens, previous_states):
        return tf.gather(self.table, tf.reshape(previous_tokens, [-1])), previous_states

    def compute_output_shape(self, tokens_shape):
        return (*tokens_shape, self.vocab_size)

    def get_config(self):
        return {
            **super().get_config(),
            "vocab_size": self.vocab_size,
            "blank": self.blank,
            "interpolation": self.interpolation,
            "delta": self.delta,
        }
