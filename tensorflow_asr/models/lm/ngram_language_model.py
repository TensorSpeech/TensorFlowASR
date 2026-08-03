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
"""GPU-resident n-gram language model, held as flat sorted tensors -- NGPU-LM"""

import gzip
import logging
from collections import defaultdict

import numpy as np

from tensorflow_asr import keras, tf
from tensorflow_asr.models.lm.language_model import LanguageModel

logger = logging.getLogger(__name__)

# Sits past every real key, so a query that runs off the end of the arc table lands on padding and
# compares unequal instead of matching some unrelated arc. Any value above `max_states * V` works;
# this one is far above it and still nowhere near the int64 ceiling.
PAD_KEY = np.int64(1) << np.int64(62)


def count_ngrams(token_sequences, order: int, vocab_size: int, blank: int):
    """
    Count n-grams of every order from 1 to `order` over tokenized transcripts.

    Returns one dict per **context length** `k` in `0 .. order - 1`, mapping a context tuple to its
    successor counts: `counts[k][ctx][w]` is how often `w` followed `ctx`. `counts[0][()]` is
    therefore the unigram count of each token.

    Histories are left-padded with `blank`, which doubles as start-of-sentence exactly as it does in
    `count_bigrams` and in the beam search. So the first token of an utterance is counted under the
    context `(blank, ..., blank)`, the second under `(blank, ..., blank, t1)`, and so on. That is
    what makes the model able to score the first token of an utterance at all.

    Blank is dropped from the *successor* position throughout: it is a marker, not a word, and the
    decoder never reads its column. It survives only inside contexts, as the start marker.

    Parameters
    ----------
    token_sequences : Iterable[Sequence[int]]
        Tokenized transcripts, as produced by `Tokenizer.tokenize`. Consumed once, so a generator is
        fine and is what the CLI passes.
    order : int
        Highest n-gram order. `order=4` counts unigrams through 4-grams, ie. contexts up to 3 long.
    vocab_size : int
    blank : int

    Returns
    -------
    list[dict[tuple, dict[int, int]]], length `order`
    """
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")

    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
    context_length = order - 1
    padding = (blank,) * context_length

    # `disable=False` overrides the `TQDM_DISABLE=1` that `scripts/train_lm.py` sets globally, the
    # same way every other progress bar in the repository does. This is the one pass over the whole
    # corpus and it is where a multi-GB run spends its time, so it should not be silent. No total is
    # given: `token_sequences` is a generator, and counting it first would mean reading the corpus
    # twice -- tqdm falls back to showing lines done and the rate, which is what is wanted anyway.
    for tokens in tqdm(token_sequences, desc=f"Counting n-grams up to order {order}", unit=" lines", disable=False):
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        # Anything outside the vocabulary is dropped, and so is blank: a context may contain it as
        # the start marker, but nothing may ever be predicted as blank.
        tokens = tokens[(tokens >= 0) & (tokens < vocab_size) & (tokens != blank)]
        if tokens.size == 0:
            continue
        history = padding
        for token in tokens.tolist():
            for k in range(order):
                counts[k][history[context_length - k :] if k else ()][token] += 1
            if context_length:
                history = (history + (token,))[-context_length:]
    return counts


def _continuation_counts(counts, order: int):
    """
    Replace the raw counts of every order below the highest with Kneser-Ney continuation counts.

    The Kneser-Ney insight: a lower-order estimate is only ever consulted *after* a higher-order
    context failed, so it should not answer "how often is this token seen" but "how many different
    contexts is this token seen in". The classic example is "Francisco", which is frequent but only
    ever after "San" -- a raw unigram gives it a high backoff probability it has not earned, while a
    continuation count gives it 1.

    So `adjusted[k][ctx][w]` counts the *distinct* tokens `v` for which `v + ctx` was seen followed
    by `w`, which is exactly what falls out of walking the level above. The highest order keeps its
    raw counts, since nothing ever backs off *into* it.
    """
    adjusted = [None] * order
    adjusted[order - 1] = counts[order - 1]
    for k in range(order - 2, -1, -1):
        continuation = defaultdict(lambda: defaultdict(int))
        for context, successors in counts[k + 1].items():
            shorter = context[1:]
            for token in successors:
                continuation[shorter][token] += 1
        adjusted[k] = continuation
    return adjusted


def _kneser_ney_probs(adjusted, order: int, vocab_size: int, blank: int, discount: float):
    """
    Interpolated Kneser-Ney, built bottom-up.

    At each level, mass `discount` is taken off every seen count and redistributed over the
    lower-order estimate:

        p_k(w | ctx) = max(c_k(ctx, w) - D, 0) / N_k(ctx)  +  gamma_k(ctx) * p_{k-1}(w | ctx[1:])
        gamma_k(ctx) = D * |{w : c_k(ctx, w) > 0}| / N_k(ctx)

    with `N_k(ctx)` the total count of the context. The unigram level interpolates with the uniform
    distribution over labels instead, which is what guarantees **every label keeps some mass** --
    the same reason `build_table` insists on `delta > 0` for the bigram. A single zero here would
    become `ln 0 = -inf` and kill an otherwise legal hypothesis in the beam.

    Returns one dict per context length, mapping context to `{token: probability}`. Only *observed*
    successors are stored, except at level 0 which stores every label -- see `build_arcs` for why
    that matters.
    """
    labels = [w for w in range(vocab_size) if w != blank]
    num_labels = len(labels)
    if num_labels == 0:
        raise ValueError(f"vocab_size={vocab_size} leaves no labels once blank={blank} is excluded")

    probs = [dict() for _ in range(order)]

    unigram_counts = adjusted[0].get((), {})
    total = float(sum(unigram_counts.values()))
    if total <= 0:
        raise ValueError("No n-grams were counted -- the LM training text produced no usable tokens")
    seen = len(unigram_counts)
    # Every label gets at least this, so no label is ever impossible.
    uniform_share = (discount * seen / total) / num_labels
    probs[0][()] = {w: max(unigram_counts.get(w, 0) - discount, 0.0) / total + uniform_share for w in labels}

    for k in range(1, order):
        level = {}
        for context, successors in adjusted[k].items():
            total = float(sum(successors.values()))
            if total <= 0:
                continue
            gamma = discount * len(successors) / total
            shorter = context[1:]
            level[context] = {w: max(count - discount, 0.0) / total + gamma * _walk(probs, k - 1, shorter, w) for w, count in successors.items()}
        probs[k] = level
    return probs


def _walk(probs, k: int, context, token: int) -> float:
    """
    Read `p(token | context)` off the levels built so far, backing off until something answers.

    Level 0 holds every label, so this always terminates with a real number.
    """
    while k > 0:
        level = probs[k].get(context)
        if level is not None and token in level:
            return level[token]
        k -= 1
        context = context[1:]
    return probs[0][()][token]


def build_arcs(
    counts,
    order: int,
    vocab_size: int,
    blank: int,
    discount: float = 0.75,
    min_counts=None,
    max_arcs: int = 1_000_000,
    max_states: int = 200_000,
):
    """
    Turn n-gram counts into the flat sorted tensors `NGramLanguageModel` scores from.

    This is the data structure of NGPU-LM (https://arxiv.org/abs/2505.22857): the trie is flattened
    into arrays so a GPU can answer a whole batch of lookups with an index-select instead of chasing
    pointers. A **state** is a context; an **arc** is a `(state, token)` pair with an explicit
    probability; a state's **backoff** link points at the same context with its oldest token dropped.

    The one departure from the paper is the arc index. The paper sorts arcs by `(from_state, token)`
    and keeps `start_arcs` / `end_arcs` to delimit each state's block. Packing the same sort key into
    a single integer, `from_state * vocab_size + token`, makes the whole table one sorted array that
    a single binary search can address -- so the per-state ranges are not needed at all, and the
    lookup never needs a ragged slice. That matters here because the decoder must stay convertible
    to XLA and TFLite, which want static shapes.

    Three invariants make the lookup in `NGramLanguageModel.call_next` correct:

    1. **The unigram state holds every label.** Backing off always terminates with a hit, so no
       token is ever unscored and there is no `<unk>` special case -- as in the paper.
    2. **Every state's backoff target is itself a state.** The set of contexts is closed under
       dropping the oldest token, adding empty states where pruning removed all of a context's arcs.
    3. **Every state is a distribution over the labels.** Backoff weights are recomputed *after*
       pruning from the normalisation identity below, not carried over from the smoothing, so a
       pruned model is still a proper LM rather than one that quietly leaks mass.

    On (3): for a state whose surviving arcs are `A`, the backoff weight is

        gamma(ctx) = (1 - sum_{w in A} p(w | ctx)) / (1 - sum_{w in A} p(w | ctx[1:]))

    -- the mass not spoken for by the explicit arcs, divided by the mass the backoff distribution
    assigns to the same leftovers. This is the standard ARPA backoff weight, and computing it bottom
    up means each level normalises against the already-final level below it.

    Parameters
    ----------
    counts : list[dict]
        As returned by `count_ngrams`.
    order : int
    vocab_size : int
    blank : int
    discount : float
        Kneser-Ney discount `D`, in (0, 1). Must be > 0, otherwise the unigram level keeps no
        uniform floor and an unseen label lands on `ln 0 = -inf`.
    min_counts : Optional[Sequence[int]]
        Per-context-length count cutoff: an arc at context length `k` is kept only if its **raw**
        count is at least `min_counts[k]`. Defaults to keeping everything. Cutoffs read off raw
        counts rather than the Kneser-Ney continuation counts because "seen at least twice" is a
        statement about the data, not about the smoothing.
    max_arcs : int
        Hard cap on the arc table. Once cutoffs are applied, the lowest-count arcs are dropped until
        the table fits. Must leave room for the unigram state's full label set.
    max_states : int
        Hard cap on the state table.

    Returns
    -------
    dict of numpy arrays: `arc_keys` [max_arcs] int64, `arc_weights` [max_arcs] float32,
    `arc_to_states` [max_arcs] int32, `backoff_weights` [max_states] float32,
    `backoff_to_states` [max_states] int32, plus `bos_state` and a `stats` dict.
    """
    if not 0.0 < discount < 1.0:
        raise ValueError(f"discount must be in (0, 1), got {discount}. At 0 an unseen label gets ln 0 = -inf.")
    num_labels = vocab_size - (1 if 0 <= blank < vocab_size else 0)
    if max_arcs < num_labels:
        raise ValueError(f"max_arcs={max_arcs} cannot hold the unigram state's {num_labels} labels, which are mandatory")
    if min_counts is None:
        min_counts = [1] * order
    if len(min_counts) != order:
        raise ValueError(f"min_counts must have one cutoff per context length, ie. {order} entries, got {len(min_counts)}")

    adjusted = _continuation_counts(counts, order)
    probs = _kneser_ney_probs(adjusted, order, vocab_size, blank, discount)

    # ---------------------------------------------------------------- choose which arcs survive
    # Level 0 is mandatory and is not a candidate for pruning: it is what makes backoff terminate.
    candidates = []  # (raw count, context length, context, token)
    for k in range(1, order):
        cutoff = int(min_counts[k])
        for context, successors in probs[k].items():
            raw = counts[k].get(context, {})
            for token in successors:
                count = raw.get(token, 0)
                if count >= cutoff:
                    candidates.append((count, k, context, token))

    budget = max_arcs - num_labels
    if len(candidates) > budget:
        # Sort by count descending, then by (length, context, token) so a tie breaks the same way on
        # every run -- a model that reshuffles between fits would be impossible to reproduce.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        logger.warning(
            f"Pruning {len(candidates) - budget} of {len(candidates)} arcs to fit max_arcs={max_arcs}. "
            f"Raise max_arcs, or raise min_counts to choose which arcs go rather than letting the budget decide."
        )
        candidates = candidates[:budget]

    kept = defaultdict(set)  # context -> set of tokens
    for _, _, context, token in candidates:
        kept[context].add(token)

    levels = [dict() for _ in range(order)]
    levels[0][()] = probs[0][()]
    for context, tokens in kept.items():
        levels[len(context)][context] = {token: probs[len(context)][context][token] for token in tokens}

    # Padded start context, matching how `count_ngrams` padded the histories it counted.
    return _assemble(levels, order, vocab_size, blank, (blank,) * (order - 1), max_arcs, max_states)


def _assemble(levels, order: int, vocab_size: int, blank: int, bos_context, max_arcs: int, max_states: int):
    """
    Turn per-context explicit probabilities into the flat sorted tensors the lookup reads.

    Shared by both ways of building this model -- counting (`build_arcs`) and reading an ARPA file
    (`read_arpa`) -- because everything after "which n-grams do we keep, and what is each one's
    probability" is identical between them.

    `levels[k]` maps a context of length `k` to `{token: probability}`, holding only the n-grams that
    survive. `levels[0][()]` is special: it must hold **every** label, because it is where the backoff
    walk terminates.

    Backoff weights are **recomputed here rather than taken from the input**, from

        gamma(ctx) = (1 - sum_{w in A} p(w | ctx)) / (1 - sum_{w in A} p(w | ctx[1:]))

    where `A` is the context's surviving arcs. This is the standard ARPA backoff weight, computed
    bottom up so each level normalises against the already-final level below it. Recomputing is what
    makes both callers safe: pruning drops arcs, and the ARPA reader drops `</s>`, and either would
    otherwise leave states quietly leaking probability mass.
    """
    # Closed under backoff: a surviving context drags its every suffix in with it, even one that kept
    # no arcs of its own, because the backoff walk has to pass through it.
    contexts = {()}
    for level in levels:
        for context in level:
            for i in range(len(context) + 1):
                contexts.add(context[i:])
    # The start-of-sentence context, so an utterance can be scored from its first token.
    for i in range(len(bos_context) + 1):
        contexts.add(bos_context[i:])

    ordered = sorted(contexts, key=lambda context: (len(context), context))
    if len(ordered) > max_states:
        raise ValueError(
            f"{len(ordered)} states exceed max_states={max_states}. Raise max_states, or raise min_counts / lower max_arcs so fewer contexts survive."
        )
    state_ids = {context: index for index, context in enumerate(ordered)}

    def longest_state(context):
        """The longest suffix of `context` that is a state. `()` always is, so this terminates."""
        while context not in state_ids:
            context = context[1:]
        return state_ids[context]

    # ---------------------------------------------------------------- final probabilities
    # Bottom-up, so each level normalises against the level below it in its already-pruned form.
    final = [dict() for _ in range(order)]
    backoff = np.zeros([len(ordered)], dtype=np.float64)
    final[0][()] = levels[0][()]

    def final_prob(k: int, context, token: int) -> float:
        """`p(token | context)` under the pruned model, following backoff weights."""
        total = 0.0
        while k > 0:
            level = final[k].get(context)
            if level is not None and token in level:
                return float(np.exp(total)) * level[token]
            total += backoff[state_ids[context]] if context in state_ids else 0.0
            k -= 1
            context = context[1:]
        return float(np.exp(total)) * final[0][()][token]

    for context in ordered:
        k = len(context)
        if k == 0:
            continue
        level = levels[k].get(context, {})
        explicit = sum(level.values())
        lower = sum(final_prob(k - 1, context[1:], token) for token in level)
        # `leftover` is the mass the level below puts on the tokens this context does *not* list --
        # the only place backoff can deposit anything.
        leftover = 1.0 - lower
        if leftover > 1e-9 and explicit < 1.0:
            # The ordinary case: hand the unclaimed mass to backoff, which makes the state sum to
            # exactly one since `explicit + gamma * leftover == 1` by construction.
            backoff[state_ids[context]] = np.log((1.0 - explicit) / leftover)
        else:
            # This context already lists every token the level below can reach, so backoff has
            # nowhere to put the remainder and adjusting gamma cannot fix the total. The arcs have to
            # be a distribution on their own. Without this a state whose arcs sum to less than one --
            # normal for an ARPA, where `</s>` and the discount both hold mass back -- would stay
            # sub-normalised, and no amount of backoff weight would show it up.
            if explicit > 0.0:
                level = {token: probability / explicit for token, probability in level.items()}
            backoff[state_ids[context]] = np.log(1e-10)  # unreachable, but keep it finite
        final[k][context] = level

    # ---------------------------------------------------------------- flatten
    arc_keys, arc_weights, arc_to_states = [], [], []
    for context in ordered:
        state = state_ids[context]
        k = len(context)
        level = final[0][()] if k == 0 else final[k].get(context, {})
        for token in sorted(level):
            arc_keys.append(state * vocab_size + token)
            arc_weights.append(np.log(level[token]))
            arc_to_states.append(longest_state((context + (token,))[-(order - 1) :] if order > 1 else ()))

    num_arcs = len(arc_keys)
    if num_arcs > max_arcs:  # unreachable given the budget above, but the arrays below assume it
        raise ValueError(f"{num_arcs} arcs exceed max_arcs={max_arcs}")

    keys = np.full([max_arcs], PAD_KEY, dtype=np.int64)
    weights = np.zeros([max_arcs], dtype=np.float32)
    destinations = np.zeros([max_arcs], dtype=np.int32)
    if num_arcs:
        # Sorted by construction -- states are walked in id order and tokens in ascending order
        # within each -- but sorting is cheap and makes the invariant the lookup relies on explicit.
        permutation = np.argsort(np.asarray(arc_keys, dtype=np.int64), kind="stable")
        keys[:num_arcs] = np.asarray(arc_keys, dtype=np.int64)[permutation]
        weights[:num_arcs] = np.asarray(arc_weights, dtype=np.float32)[permutation]
        destinations[:num_arcs] = np.asarray(arc_to_states, dtype=np.int32)[permutation]

    backoff_weights = np.zeros([max_states], dtype=np.float32)
    backoff_weights[: len(ordered)] = backoff.astype(np.float32)
    backoff_to_states = np.zeros([max_states], dtype=np.int32)
    for context in ordered:
        backoff_to_states[state_ids[context]] = state_ids[context[1:]] if context else 0

    return {
        "arc_keys": keys,
        "arc_weights": weights,
        "arc_to_states": destinations,
        "backoff_weights": backoff_weights,
        "backoff_to_states": backoff_to_states,
        "bos_state": state_ids[bos_context],
        "stats": {
            "order": order,
            "arcs": num_arcs,
            "arcs_budget": max_arcs,
            "states": len(ordered),
            "states_budget": max_states,
            "arcs_per_order": {k: sum(len(level) for level in levels[k].values()) for k in range(order)},
        },
    }


ARPA_BOS = "<s>"
ARPA_EOS = "</s>"
ARPA_UNK = "<unk>"


def _arpa_lines(path):
    """Stream the lines of an ARPA file, transparently un-gzipping one written as `.gz`."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def read_arpa(
    path,
    vocab_size: int,
    blank: int,
    order: int = None,
    max_arcs: int = 1_000_000,
    max_states: int = 200_000,
    token_of=None,
):
    """
    Read an ARPA n-gram file -- the format KenLM's `lmplz` writes -- into the arc tensors.

    This is the path NGPU-LM itself takes (https://arxiv.org/abs/2505.22857): the paper builds its
    models with KenLM and converts the ARPA. Use it instead of `build_arcs` whenever the corpus is
    too big to count in this process. `count_ngrams` holds every n-gram in python dicts, which costs
    roughly 300-400 bytes per token and so tops out around a few million tokens; `lmplz` does a
    disk-based merge sort in bounded memory and handles corpora three orders of magnitude larger.

    **The ARPA must be token-level, over this tokenizer's integer ids.** `call_next` returns
    log-probabilities indexed against the transducer's vocabulary, so a word-level ARPA cannot be
    used, however good it is. `tensorflow_asr utils create_lm_text` writes a corpus in the form
    `lmplz` needs -- one sentence per line, tokens as space-separated integers.

    Details of the conversion:

    - **ARPA is log base 10**; the arc tensors are natural log, so every weight is converted.
    - **`<s>` is the start marker**, and maps to `blank`, which plays that role everywhere else here
      (see `count_ngrams`). It appears only inside contexts, never as a prediction.
    - **`</s>` is conditioned away, not merely dropped.** This repository has no end-of-sentence
      symbol -- a hypothesis ends when the encoder frames run out, never on an emitted token (see
      `LanguageModel`) -- so an n-gram predicting it has nothing to say here. But deleting those
      entries would leak their mass: a context that listed every label explicitly would then sum to
      `1 - p(</s> | ctx)` with no backoff path to make up the difference, which is a silently
      sub-normalised LM. Each context's remaining arcs are therefore rescaled by
      `1 / (1 - p(</s> | ctx))`, ie. the model is conditioned on "this sentence does not end here",
      which is the honest reading of an LM with no end symbol.
    - **`<unk>` becomes the floor** for any label the ARPA never mentions, which is what the paper
      does ("if the token is absent in ARPA, we use normalized `<unk>` weight"). Without it those
      labels would have no probability at all and the beam would see `ln 0 = -inf`.
    - **Backoff weights are recomputed, not read.** Dropping `</s>` and adding the `<unk>` floor both
      move probability mass, so the file's own backoff weights no longer normalise. `_assemble`
      recomputes them from the mass actually left over, which restores the invariant the decoder
      needs. The unigram level is renormalised over the labels for the same reason.

    Parameters
    ----------
    path : str
        ARPA file, optionally gzipped.
    vocab_size : int
    blank : int
    order : Optional[int]
        Truncate to this order. Defaults to the ARPA's own, and cannot exceed it.
    max_arcs, max_states : int
        Tensor shapes, as in `build_arcs`. An ARPA that does not fit raises -- prune it at build
        time with `lmplz --prune` instead, which is cheaper and better informed than dropping arcs
        here.
    token_of : Optional[Callable[[str], int]]
        Maps an ARPA word to a token id. Defaults to `int`, ie. the file's words *are* the ids. Pass
        something else to read an ARPA written over token strings.

    Returns
    -------
    dict of numpy arrays, exactly as `build_arcs` returns.
    """
    if token_of is None:
        token_of = int  # a token-level ARPA spells each token as its integer id

    def to_token(word):
        """ARPA word -> a usable label id, or None if it cannot be one."""
        try:
            token = int(token_of(word))
        except (TypeError, ValueError):
            return None
        # `blank` is excluded deliberately: it is the start marker, never something to predict.
        return token if 0 <= token < vocab_size and token != blank else None

    header = {}
    unigrams = {}
    explicit = defaultdict(dict)  # context tuple -> {token: probability}
    eos_mass = {}  # context tuple -> p(</s> | context), conditioned away below
    unk_prob = None
    section = None
    skipped = 0
    saw_bos = False

    for raw in _arpa_lines(path):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("\\"):
            if line.endswith("-grams:"):
                section = int(line[1:].split("-", 1)[0])
            elif line == "\\end\\":
                break
            else:  # \data\
                section = None
            continue
        if section is None:
            if line.startswith("ngram"):
                left, _, right = line[len("ngram") :].strip().partition("=")
                try:
                    header[int(left)] = int(right)
                except ValueError:
                    pass
            continue

        # `logprob <tab> w1 ... wk [<tab> backoff]`. Splitting on whitespace rather than tabs
        # handles the space-separated files some tools emit; `section` says how many words to take,
        # so the optional trailing backoff is unambiguous and simply ignored -- see the docstring.
        parts = line.split()
        if len(parts) < 1 + section:
            skipped += 1
            continue
        try:
            probability = 10.0 ** float(parts[0])
        except ValueError:
            skipped += 1
            continue
        words = parts[1 : 1 + section]

        # The context is resolved first, because `</s>`'s probability has to be attributed to the
        # context it was predicted from before it can be conditioned away.
        context = []
        for word in words[:-1]:
            if word == ARPA_BOS:
                context.append(blank)
                saw_bos = True
                continue
            mapped = to_token(word)
            if mapped is None:  # `</s>`, `<unk>` or an unmappable word cannot be part of a context
                context = None
                break
            context.append(mapped)
        if context is None:
            skipped += 1
            continue
        context = tuple(context)

        target = words[-1]
        if target == ARPA_EOS:
            eos_mass[context] = eos_mass.get(context, 0.0) + probability
            continue
        if target == ARPA_BOS:
            continue  # never predicted
        if target.lower() == ARPA_UNK:
            if section == 1:
                unk_prob = probability
            continue

        token = to_token(target)
        if token is None:
            skipped += 1
            continue

        if section == 1:
            unigrams[token] = probability
        else:
            explicit[context][token] = probability

    if not unigrams:
        raise ValueError(f"No usable unigrams in {path}. Is it a token-level ARPA, with tokens written as integer ids?")

    arpa_order = max(header) if header else max((len(context) + 1 for context in explicit), default=1)
    if order is None:
        order = arpa_order
    elif order > arpa_order:
        logger.warning(f"Requested order {order} exceeds the ARPA's {arpa_order}; using {arpa_order}")
        order = arpa_order

    labels = [w for w in range(vocab_size) if w != blank]
    if unk_prob is None:
        # Nothing to floor unseen labels with. The rarest thing the file does mention is the closest
        # honest stand-in, and it keeps every label finite, which is the invariant that matters.
        unk_prob = min(unigrams.values())
        logger.warning(f"{path} has no <unk>; flooring the {len(labels) - len(unigrams)} unlisted labels at the rarest unigram instead")
    if not saw_bos:
        logger.warning(f"{path} has no {ARPA_BOS} n-grams, so the first token of an utterance will be scored out of context")

    # Renormalise over the labels: `<s>`, `</s>` and `<unk>` all held mass in the file and none of
    # them survive here, so what is left has to be made a distribution again.
    # Condition on "the sentence does not end here", removing `</s>` without leaking its mass. The
    # unigram level gets the same treatment for free: renormalising over the labels below divides
    # out `</s>`, `<s>` and `<unk>` together.
    floored = {w: unigrams.get(w, unk_prob) for w in labels}
    total = sum(floored.values())
    levels = [dict() for _ in range(order)]
    levels[0][()] = {w: p / total for w, p in floored.items()}
    for context, successors in explicit.items():
        if not 1 <= len(context) <= order - 1:
            continue
        removed = eos_mass.get(context, 0.0)
        if removed > 0.0:
            scale = 1.0 / max(1.0 - removed, 1e-9)
            successors = {w: p * scale for w, p in successors.items()}
        levels[len(context)][context] = successors

    # A well-formed ARPA gives each context a probability mass of at most one; `_assemble` then hands
    # the leftover to backoff. If a context is already over one there is no leftover to give, the
    # clamp there takes over, and the state ends up summing to more than one -- an LM that quietly
    # wrecks decoding rather than failing. The usual cause is that the file is not the token-level
    # ARPA this expects: a word-level model, or one built over a different tokenizer, collides many
    # distinct words onto the same id and stacks their probabilities.
    overfull = [(sum(successors.values()), context) for context in levels[0] for successors in [levels[0][context]]]
    overfull += [(sum(successors.values()), context) for level in levels[1:] for context, successors in level.items()]
    worst = max((entry for entry in overfull if entry[0] > 1.01), default=None)
    if worst is not None:
        count = sum(1 for mass, _ in overfull if mass > 1.01)
        raise ValueError(
            f"{count} of {len(overfull)} contexts in {path} carry more probability than one "
            f"(worst: {worst[0]:.3f} for context {worst[1]}). This is not a well-formed token-level ARPA -- "
            f"the usual cause is a word-level model, or one built over a different tokenizer, so several "
            f"words collapse onto the same token id. Rebuild it with `tensorflow_asr utils create_lm_text`."
        )

    if max_arcs < len(labels):
        raise ValueError(f"max_arcs={max_arcs} cannot hold the unigram state's {len(labels)} labels, which are mandatory")

    arrays = _assemble(levels, order, vocab_size, blank, (blank,) * min(1, order - 1), max_arcs, max_states)
    arrays["stats"].update(source="arpa", arpa_order=arpa_order, skipped_lines=skipped)
    return arrays


@keras.utils.register_keras_serializable(package=__name__)
class NGramLanguageModel(LanguageModel):
    """
    N-gram language model over the transducer's own vocabulary, held as flat GPU tensors.

    Ref: "NGPU-LM: GPU-Accelerated N-Gram Language Model for Context-Biasing in Greedy ASR
         Decoding", V. Bataev et al., Interspeech 2025, https://arxiv.org/abs/2505.22857

    `BigramLanguageModel` is the same idea at order 2, where a dense `[V, V]` table is small enough
    to store outright. That stops working immediately: order 4 over a 1000-token vocabulary would be
    a `[V, V, V, V]` table of 10^12 entries, of which a real corpus fills a vanishing fraction. This
    model stores only the n-grams that were actually seen, as a sorted arc table, and reaches the
    rest by backing off.

    **The vocabulary must be the transducer's own**, the same constraint `LSTMLanguageModel`
    carries: `call_next` returns log-probabilities indexed against it, so the counts have to come
    from text tokenized by the same tokenizer. A word-level n-gram from another toolkit cannot be
    dropped in.

    Fit it with `tensorflow_asr train_lm`, which dispatches to `fit_counts` rather than gradient
    descent -- an n-gram's estimate is a ratio of counts, exact in one pass. Use `--target=external`
    for the LM that gets fused in, and point it at as much target-domain text as you have.

    Sizing
    ------
    `max_arcs` and `max_states` are the tensor shapes, so they are fixed at construction and live in
    the config. They are a **pruning budget**, not an allocation guess: fitting keeps the
    highest-count n-grams that fit and drops the rest, which is what the paper does for its larger
    models and what LODR does at order 2. Memory is `max_arcs * 16 + max_states * 8` bytes -- 16 MB
    per million arcs -- so the defaults sit around 17 MB, and the paper's models are all under
    100 MB. `fit_counts` logs actual usage against the budget so it can be tightened.

    Decoding state
    --------------
    State is the `[B, V]` table of next-state ids that the lookup produces anyway, so a step costs
    one pass rather than one to advance and another to score. The next step recovers its state with
    `state[b, previous_token]`. That satisfies the beam's requirement of a single tensor batched on
    axis 0: `_select_beam_states` re-orders it on hypothesis recombination exactly as it does the
    prediction network's, with no special casing. It costs `B * W * V` int32 -- about 1 MB at 256
    hypotheses over a 1000-token vocabulary.

    Blank is inert. It is never a successor, its score column is exactly 0.0 (matching
    `BigramLanguageModel`, so `lm_alpha` and `lm_beta` mean the same thing across both), and its
    state column is a self-loop, so a hypothesis that emits blank keeps the context it had.
    """

    def __init__(
        self,
        vocab_size: int,
        blank: int = 0,
        order: int = 6,
        max_arcs: int = 20_000_000,
        max_states: int = 10_000_000,
        discount: float = 0.75,
        # Deliberately `None` (= keep every n-gram) rather than a cutoff list. A list here would have
        # to be as long as `order`, so any concrete default silently locks the class to one order --
        # `[1, 1, 2, 2]` makes `order=3` raise. Set cutoffs in the config, next to the `order` they
        # have to agree with. (A list default would also be a shared mutable, the usual trap.)
        min_counts=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        self.vocab_size = vocab_size
        self.blank = blank
        self.order = order
        self.max_arcs = max_arcs
        self.max_states = max_states
        self.discount = discount
        self.min_counts = list(min_counts) if min_counts is not None else None

        # Created here rather than in `build` because the decoder calls this model from inside a
        # `tf.while_loop`, where creating variables fails. Non-trainable throughout: counting sets
        # these, and no gradient should ever move them.
        #
        # `arc_keys` starts at the padding sentinel rather than zero so an *unfitted* model is
        # well-defined -- every lookup misses, every score is 0.0. Zeros would instead look like a
        # real arc from state 0 on token 0.
        self.arc_keys = self.add_weight(
            name="arc_keys",
            shape=[max_arcs],
            initializer=keras.initializers.Constant(PAD_KEY),
            trainable=False,
            dtype="int64",
        )
        self.arc_weights = self.add_weight(name="arc_weights", shape=[max_arcs], initializer="zeros", trainable=False, dtype="float32")
        self.arc_to_states = self.add_weight(name="arc_to_states", shape=[max_arcs], initializer="zeros", trainable=False, dtype="int32")
        self.backoff_weights = self.add_weight(name="backoff_weights", shape=[max_states], initializer="zeros", trainable=False, dtype="float32")
        self.backoff_to_states = self.add_weight(name="backoff_to_states", shape=[max_states], initializer="zeros", trainable=False, dtype="int32")
        # Which state an utterance starts in. A weight rather than a python attribute because it is
        # found by fitting, so it has to travel in the h5 alongside the tables.
        self.bos_state = self.add_weight(name="bos_state", shape=[], initializer="zeros", trainable=False, dtype="int32")
        # Every weight this model will ever have now exists, so it really is built. Saying so lets
        # `save_weights` / `load_weights` work without a dummy forward pass first, which keras
        # otherwise insists on.
        self.built = True

    def fit_counts(self, token_sequences):
        """
        Fit by counting, the closed-form maximum likelihood estimate. Returns a stats dict.

        `train_lm.py` calls this instead of `fit` when the model provides it, which is how the same
        script trains both a neural LM by gradient descent and this one by a single pass.
        """
        counts = count_ngrams(token_sequences, order=self.order, vocab_size=self.vocab_size, blank=self.blank)
        arrays = build_arcs(
            counts,
            order=self.order,
            vocab_size=self.vocab_size,
            blank=self.blank,
            discount=self.discount,
            min_counts=self.min_counts,
            max_arcs=self.max_arcs,
            max_states=self.max_states,
        )
        return self._assign(arrays)

    def load_arpa(self, path):
        """
        Fill the tables from an ARPA file built elsewhere, typically by KenLM's `lmplz`.

        The alternative to `fit_counts`, and the only one that scales: counting in this process is
        bounded by python dict memory, `lmplz` is not. `train_lm --arpa=<path>` calls this. See
        `read_arpa` for what the conversion does and what the ARPA has to look like.
        """
        arrays = read_arpa(
            path,
            vocab_size=self.vocab_size,
            blank=self.blank,
            order=self.order,
            max_arcs=self.max_arcs,
            max_states=self.max_states,
        )
        # An ARPA of a different order than the config asked for is worth saying out loud: `order`
        # is a config value that has to survive into the h5 reload, and the tables now disagree with
        # it. Everything still works -- the lookup unrolls `self.order` times, and extra iterations
        # on an already-terminated backoff walk are a no-op -- but the config should be corrected.
        if arrays["stats"]["order"] != self.order:
            logger.warning(f"Config says order={self.order} but the ARPA gave order={arrays['stats']['order']}; update the config to match")
        return self._assign(arrays)

    def _assign(self, arrays):
        """Copy built tables into the weights. Shared by `fit_counts` and `load_arpa`."""
        self.arc_keys.assign(arrays["arc_keys"])
        self.arc_weights.assign(arrays["arc_weights"])
        self.arc_to_states.assign(arrays["arc_to_states"])
        self.backoff_weights.assign(arrays["backoff_weights"])
        self.backoff_to_states.assign(arrays["backoff_to_states"])
        self.bos_state.assign(arrays["bos_state"])
        return arrays["stats"]

    def get_initial_state(self, batch_size):
        # State is a *next-state table*, indexed by the token that `call_next` will be handed as
        # `previous_tokens`. So the initial state is the transitions out of the start state, not a
        # constant: a hypothesis whose first call reports token `t` must land in the state reached by
        # consuming `t` from the start, exactly as it would on any later step. Column blank is the
        # self-loop `_step` writes, which covers the first call of all -- nothing emitted yet.
        _, next_states = self._step(tf.fill([batch_size], tf.cast(self.bos_state, tf.int32)))
        return next_states

    def _step(self, states):
        """
        Score every label from a batch of states, and say where each one leads. Algorithm 1 of the
        paper, with the loop unrolled.

        The paper's loop runs "while some token is still unscored", bounded by the n-gram order
        because that is how long a backoff chain can be. Unrolling it to exactly `order` iterations
        trades a few wasted comparisons on the last pass for a graph with no data-dependent
        condition -- which is what keeps this convertible to XLA and TFLite, and lets the whole thing
        run without a single synchronisation back to the host. The paper needed a custom Triton
        kernel to get the same effect from a dynamic loop.

        `states` has any shape `S`; the results are `S + [V]`.
        """
        shape = tf.shape(states)
        current = tf.reshape(states, [-1])  # [M]
        entry = current
        rows = tf.shape(current)[0]
        tokens = tf.range(self.vocab_size, dtype=tf.int64)  # [V]

        scores = tf.zeros([rows, self.vocab_size], dtype=tf.float32)
        # An unresolved token keeps its state, which cannot happen -- the unigram state holds every
        # label -- but makes the initial value meaningful rather than arbitrary.
        next_states = tf.tile(tf.expand_dims(entry, axis=1), [1, self.vocab_size])
        found = tf.zeros([rows, self.vocab_size], dtype=tf.bool)
        accumulated = tf.zeros([rows, 1], dtype=tf.float32)

        sorted_keys = tf.expand_dims(self.arc_keys, axis=0)  # [1, max_arcs]
        for _ in range(self.order):
            keys = tf.expand_dims(tf.cast(current, tf.int64) * self.vocab_size, axis=1) + tf.expand_dims(tokens, axis=0)  # [M, V]
            # One binary search over the whole table. `searchsorted` wants matching leading
            # dimensions, hence the flatten to a single row and back.
            indices = tf.searchsorted(sorted_keys, tf.reshape(keys, [1, -1]), side="left")
            indices = tf.reshape(indices, [rows, self.vocab_size])
            indices = tf.minimum(indices, self.max_arcs - 1)  # a key past the last arc would index off the end
            # `searchsorted` gives the insertion point; the arc exists only if what sits there is
            # the key we asked for. Padding is the sentinel, so it never compares equal.
            hit = tf.equal(tf.gather(self.arc_keys, indices), keys)
            fresh = tf.logical_and(hit, tf.logical_not(found))

            scores = tf.where(fresh, accumulated + tf.gather(self.arc_weights, indices), scores)
            next_states = tf.where(fresh, tf.gather(self.arc_to_states, indices), next_states)
            found = tf.logical_or(found, fresh)

            accumulated = accumulated + tf.expand_dims(tf.gather(self.backoff_weights, current), axis=1)
            current = tf.gather(self.backoff_to_states, current)

        is_blank = tf.equal(tf.range(self.vocab_size, dtype=tf.int32), self.blank)  # [V]
        # Blank is not a word: it scores 0.0 so the fusion leaves it alone (matching
        # `BigramLanguageModel` and `_internal_lm_log_probs`), and it loops back to the state it came
        # from so emitting one does not change the context.
        scores = tf.where(is_blank, tf.zeros_like(scores), scores)
        next_states = tf.where(is_blank, tf.expand_dims(entry, axis=1), next_states)

        trailing = tf.concat([shape, [self.vocab_size]], axis=0)
        return tf.reshape(scores, trailing), tf.reshape(next_states, trailing)

    def call(self, tokens, training=False):
        """
        Teacher-forced scoring of whole sequences, `[B, U]` -> `[B, U, V]`.

        `tokens` is already shifted, so position `u` holds the token *before* the one to predict, and
        the context of position `u` is the `order - 1` entries ending at `u`.

        There is no recurrence to unroll here. An n-gram's state depends only on a bounded window of
        the input, so every position's state is reachable in `order - 1` steps taken over the whole
        `[B, U]` grid at once -- no `tf.while_loop`, no scan, and no dependence on `U`.
        """
        tokens = tf.cast(tokens, tf.int32)
        batch_size, length = tf.shape(tokens)[0], tf.shape(tokens)[1]
        states = tf.fill([batch_size, length], tf.cast(self.bos_state, tf.int32))
        for offset in range(self.order - 2, -1, -1):
            # The token `offset` positions back, with the start of the sequence padded by blank --
            # which `_step` treats as a self-loop, so a short prefix simply stays in the start state.
            shifted = tf.concat([tf.fill([batch_size, offset], tf.cast(self.blank, tf.int32)), tokens[:, : length - offset]], axis=1)
            _, advanced = self._step(states)
            states = tf.gather(advanced, shifted, batch_dims=2)
        scores, _ = self._step(states)
        return scores

    def call_next(self, previous_tokens, previous_states):
        # `previous_states` is the [B, V] next-state table from the previous step, so the state this
        # hypothesis is actually in is the column of the token it emitted.
        states = tf.gather(previous_states, tf.cast(previous_tokens, tf.int32), batch_dims=1)  # [B, 1]
        scores, next_states = self._step(tf.reshape(states, [-1]))  # [B, V] each
        return scores, next_states

    def compute_output_shape(self, tokens_shape):
        return (*tokens_shape, self.vocab_size)

    def get_config(self):
        return {
            **super().get_config(),
            "vocab_size": self.vocab_size,
            "blank": self.blank,
            "order": self.order,
            "max_arcs": self.max_arcs,
            "max_states": self.max_states,
            "discount": self.discount,
            "min_counts": self.min_counts,
        }
