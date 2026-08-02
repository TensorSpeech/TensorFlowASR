"""
Correctness tests for the NGPU-LM n-gram language model.

Three things have to hold. The counts must reflect the data. The model must be a proper conditional
distribution -- every state summing to one over the labels, no `-inf` anywhere, blank inert -- since
the beam adds these numbers into live hypotheses and one `-inf` kills a hypothesis outright. And the
flattened arc lookup must agree with a plain reading of the same model, because the whole point of
the data structure is that it is an implementation detail.
"""

import numpy as np
import pytest

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.ngram_language_model import NGramLanguageModel, build_arcs, count_ngrams

BLANK = 0
VOCAB = 6
ORDER = 3
CORPUS = [[1, 2, 3], [1, 2, 4], [3, 1, 2, 3], [2, 2, 1], [4, 3]]


def fit(corpus=CORPUS, order=ORDER, **kwargs):
    lm = NGramLanguageModel(vocab_size=VOCAB, blank=BLANK, order=order, max_arcs=4096, max_states=512, **kwargs)
    stats = lm.fit_counts(corpus)
    return lm, stats


def labels():
    return [w for w in range(VOCAB) if w != BLANK]


# --------------------------------------------------------------------------------------------
# a reference implementation, to check the flattened lookup against
# --------------------------------------------------------------------------------------------


def reference_scores(lm: NGramLanguageModel, context):
    """Score every label from the state `context` leads to, walking from the start state."""
    return reference_scores_from_state(lm, state_of(lm, context))


def reference_scores_from_state(lm: NGramLanguageModel, state: int):
    """
    Score every label from `state` by reading the arc tables the slow, obvious way: look for the
    arc, else add the backoff weight and shorten the context. This is the definition the flattened
    binary-search lookup in `_step` has to match.
    """
    keys = lm.arc_keys.numpy()
    weights = lm.arc_weights.numpy()
    backoff_weights = lm.backoff_weights.numpy()
    backoff_to = lm.backoff_to_states.numpy()

    scores = {}
    accumulated = 0.0
    for _ in range(lm.order):
        for token in labels():
            if token in scores:
                continue
            found = np.searchsorted(keys, state * VOCAB + token)
            if found < keys.size and keys[found] == state * VOCAB + token:
                scores[token] = accumulated + weights[found]
        accumulated += backoff_weights[state]
        state = backoff_to[state]
    return scores


def state_of(lm: NGramLanguageModel, context):
    """Walk `context` from the start state the way decoding does, one arc at a time."""
    state = int(lm.bos_state.numpy())
    keys = lm.arc_keys.numpy()
    destinations = lm.arc_to_states.numpy()
    backoff_to = lm.backoff_to_states.numpy()
    for token in context:
        walking = state
        for _ in range(lm.order):
            found = np.searchsorted(keys, walking * VOCAB + token)
            if found < keys.size and keys[found] == walking * VOCAB + token:
                state = int(destinations[found])
                break
            walking = int(backoff_to[walking])
        else:  # pragma: no cover - the unigram state holds every label, so this cannot happen
            raise AssertionError(f"no arc for token {token}, backoff did not terminate")
    return state


# --------------------------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------------------------


def test_counts_every_order_and_pads_the_start():
    counts = count_ngrams([[1, 2, 3]], order=3, vocab_size=VOCAB, blank=BLANK)
    assert len(counts) == 3
    assert counts[0][()] == {1: 1, 2: 1, 3: 1}, "level 0 is the unigram count"
    assert counts[1][(BLANK,)][1] == 1, "the first token is counted under the start marker"
    assert counts[2][(BLANK, BLANK)][1] == 1
    assert counts[1][(1,)][2] == 1 and counts[2][(BLANK, 1)][2] == 1
    assert counts[2][(1, 2)][3] == 1


def test_counts_repeats():
    counts = count_ngrams([[1, 1, 1]], order=2, vocab_size=VOCAB, blank=BLANK)
    assert counts[1][(1,)][1] == 2
    assert counts[0][()][1] == 3


def test_counts_drop_blank_and_out_of_vocabulary_successors():
    counts = count_ngrams([[1, BLANK, VOCAB + 9, 2]], order=2, vocab_size=VOCAB, blank=BLANK)
    assert counts[0][()] == {1: 1, 2: 1}, "blank is a marker, never a successor"
    assert counts[1][(1,)][2] == 1, "dropping the junk must join 1 and 2 as neighbours"


def test_counting_more_data_moves_mass():
    """Guards against a model that is well-formed but ignores the data."""
    lm, _ = fit([[1, 2]] * 50 + [[3, 4]])
    scores = reference_scores(lm, (1,))
    assert scores[2] > scores[3], "the observed successor must outrank an unobserved one"


# --------------------------------------------------------------------------------------------
# the model is a proper distribution
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_state_is_a_distribution_over_labels(order):
    lm, _ = fit(order=order)
    contexts = [(), (1,), (2,), (1, 2), (3, 1), (5,), (5, 5), (1, 5)]
    for context in contexts:
        scores = reference_scores(lm, context[: order - 1] if order > 1 else ())
        assert set(scores) == set(labels()), f"context {context} left a label unscored"
        assert np.isfinite(list(scores.values())).all(), f"context {context} produced -inf"
        total = sum(np.exp(list(scores.values())))
        np.testing.assert_allclose(total, 1.0, atol=1e-6, err_msg=f"context {context} sums to {total}")


@pytest.mark.parametrize("discount", [0.1, 0.5, 0.9])
def test_unseen_contexts_and_tokens_stay_finite(discount):
    lm, _ = fit(discount=discount)
    scores = reference_scores(lm, (5, 5))  # 5 never appears in the corpus at all
    assert np.isfinite(list(scores.values())).all()
    np.testing.assert_allclose(sum(np.exp(list(scores.values()))), 1.0, atol=1e-6)


def test_rejects_bad_discount():
    counts = count_ngrams(CORPUS, order=2, vocab_size=VOCAB, blank=BLANK)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="discount"):
            build_arcs(counts, order=2, vocab_size=VOCAB, blank=BLANK, discount=bad)


def test_rejects_a_budget_too_small_for_the_unigram_state():
    counts = count_ngrams(CORPUS, order=2, vocab_size=VOCAB, blank=BLANK)
    with pytest.raises(ValueError, match="max_arcs"):
        build_arcs(counts, order=2, vocab_size=VOCAB, blank=BLANK, max_arcs=2)


# --------------------------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------------------------


def test_pruning_to_a_budget_keeps_a_valid_model():
    """The point of recomputing backoff weights after pruning: the result is still normalised."""
    counts = count_ngrams(CORPUS, order=ORDER, vocab_size=VOCAB, blank=BLANK)
    full = build_arcs(counts, order=ORDER, vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512)
    pruned = build_arcs(counts, order=ORDER, vocab_size=VOCAB, blank=BLANK, max_arcs=len(labels()) + 3, max_states=512)
    assert pruned["stats"]["arcs"] < full["stats"]["arcs"], "the budget must actually bite"

    lm = NGramLanguageModel(vocab_size=VOCAB, blank=BLANK, order=ORDER, max_arcs=len(labels()) + 3, max_states=512)
    for name in ("arc_keys", "arc_weights", "arc_to_states", "backoff_weights", "backoff_to_states"):
        getattr(lm, name).assign(pruned[name])
    lm.bos_state.assign(pruned["bos_state"])

    for context in [(), (1,), (1, 2)]:
        scores = reference_scores(lm, context)
        np.testing.assert_allclose(sum(np.exp(list(scores.values()))), 1.0, atol=1e-6)


def test_min_counts_drops_rare_ngrams():
    counts = count_ngrams(CORPUS, order=ORDER, vocab_size=VOCAB, blank=BLANK)
    everything = build_arcs(counts, order=ORDER, vocab_size=VOCAB, blank=BLANK)
    cutoff = build_arcs(counts, order=ORDER, vocab_size=VOCAB, blank=BLANK, min_counts=[1, 2, 2])
    assert cutoff["stats"]["arcs"] < everything["stats"]["arcs"]


def test_arc_keys_are_sorted_and_padded():
    """The binary search in `_step` is only correct on a sorted table."""
    lm, stats = fit()
    keys = lm.arc_keys.numpy()
    real, padding = keys[: stats["arcs"]], keys[stats["arcs"] :]
    assert (np.diff(real) > 0).all(), "real keys must be strictly increasing -- no duplicate (state, token)"
    # Padding is one repeated sentinel, so it is sorted but not strictly, and it has to sit past
    # every real key: `searchsorted` must never place a live query before it.
    assert padding.min() > real.max(), "padding must sort past every real key"
    assert (np.diff(keys) >= 0).all(), "the whole table has to be sorted for the binary search"


def test_backoff_chain_terminates_at_the_unigram_state():
    lm, _ = fit()
    destinations = lm.backoff_to_states.numpy()
    for state in range(200):
        walking = state
        for _ in range(ORDER):
            walking = int(destinations[walking])
        assert walking == 0, f"state {state} did not reach the unigram state within {ORDER} hops"


# --------------------------------------------------------------------------------------------
# the tensorflow lookup
# --------------------------------------------------------------------------------------------


def test_step_matches_the_reference_lookup():
    """The flattened binary search must agree with reading the arcs the obvious way."""
    lm, _ = fit()
    for context in [(), (1,), (2,), (1, 2), (2, 3), (5,), (5, 5)]:
        state = state_of(lm, context)
        scores, _ = lm._step(tf.constant([state], tf.int32))  # pylint: disable=protected-access
        expected = reference_scores(lm, context)
        for token in labels():
            np.testing.assert_allclose(scores.numpy()[0, token], expected[token], rtol=1e-5, err_msg=f"context {context}, token {token}")


def test_call_next_follows_the_context():
    """Feeding tokens one at a time must land in the same state as walking the context outright."""
    lm, _ = fit()
    states = lm.get_initial_state(1)
    for step, token in enumerate([1, 2, 3]):
        scores, states = lm.call_next(tf.constant([[token]], tf.int32), states)
        assert scores.shape == (1, VOCAB)
        expected = reference_scores(lm, tuple([1, 2, 3][: step + 1]))
        for label in labels():
            np.testing.assert_allclose(scores.numpy()[0, label], expected[label], rtol=1e-5)


def test_call_next_scores_the_first_token_from_the_start_state():
    """
    Nothing has been emitted yet, so the beam hands in `previous_token = blank`. That has to score
    from the blank-padded start context, not from the unigram state -- otherwise sentence-start
    counts would be thrown away and the first token of every utterance scored out of context.
    """
    lm, _ = fit()
    scores, _ = lm.call_next(tf.constant([[BLANK]], tf.int32), lm.get_initial_state(1))
    np.testing.assert_allclose(scores.numpy()[0, 1], reference_scores(lm, ())[1], rtol=1e-5)

    unigram = reference_scores_from_state(lm, 0)
    # 1 starts two of the five training utterances, so the start state must like it more than the
    # unigram does. This is what fails if the start state collapses to the unigram state.
    assert scores.numpy()[0, 1] > unigram[1], "sentence-start counts must reach the start state"


def test_blank_is_inert():
    lm, _ = fit()
    states = lm.get_initial_state(2)
    scores, next_states = lm.call_next(tf.constant([[1], [2]], tf.int32), states)
    assert np.all(scores.numpy()[:, BLANK] == 0.0), "blank must be inert under fusion, ie. exactly 0"
    # and emitting blank must not move the context on
    after_blank = tf.gather(next_states, tf.constant([[BLANK], [BLANK]], tf.int32), batch_dims=1)
    before = tf.gather(states, tf.constant([[1], [2]], tf.int32), batch_dims=1)
    np.testing.assert_array_equal(after_blank.numpy(), before.numpy())


def test_matches_ilme_output_shape_conventions():
    """
    Fused and subtracted by the same code path as the bigram and the ILME estimate, so all three
    must agree in form: blank exactly 0, labels normalised on their own.
    """
    from tensorflow_asr.models.transducer.base_transducer import _internal_lm_log_probs

    lm, _ = fit()
    scores, _ = lm.call_next(tf.constant([[1], [2]], tf.int32), lm.get_initial_state(2))
    ilme = _internal_lm_log_probs(tf.random.stateless_normal([2, VOCAB], seed=[1, 2]), BLANK, VOCAB).numpy()

    assert np.all(scores.numpy()[:, BLANK] == 0.0) and np.all(ilme[:, BLANK] == 0.0)
    without_blank = np.delete(np.exp(scores.numpy().astype(np.float64)), BLANK, axis=-1)
    np.testing.assert_allclose(without_blank.sum(-1), 1.0, atol=1e-5)


def test_call_scores_a_whole_sequence():
    """The training path: [B, U] -> [B, U, V], and it must agree with the decoding path."""
    lm, _ = fit()
    tokens = tf.constant([[BLANK, 1, 2], [BLANK, 3, 1]], tf.int32)
    outputs = lm(tokens)  # through __call__, the way `fit` reaches it
    assert outputs.shape == (2, 3, VOCAB)

    states = lm.get_initial_state(2)
    for u in range(3):
        step, states = lm.call_next(tokens[:, u : u + 1], states)
        np.testing.assert_allclose(outputs.numpy()[:, u], step.numpy(), rtol=1e-5, err_msg=f"position {u}")


def test_call_does_not_autocast_the_token_indices():
    """Keras autocasts floating-point inputs; token indices are int32 and must come through intact."""
    lm, _ = fit()
    seen = {}
    original = lm.call

    def spy(tokens, training=False):
        seen["dtype"] = tokens.dtype
        return original(tokens, training=training)

    lm.call = spy
    lm(tf.constant([[2, 3]], tf.int32))
    assert seen["dtype"] == tf.int32, f"keras cast the indices to {seen['dtype']}"


def test_call_next_works_inside_a_while_loop():
    """
    The real constraint: the decoder calls this from inside a `tf.while_loop`. A model that built
    lazily would be creating variables there, which graph mode rejects, and a lookup with a
    data-dependent loop would not trace.
    """
    lm, _ = fit()

    @tf.function
    def loop():
        def body(i, acc, states):
            scores, states = lm.call_next(tf.fill([1, 1], i), states)
            return i + 1, acc + scores, states

        return tf.while_loop(
            lambda i, *_: i < 4,  # i = 1, 2, 3
            body,
            (tf.constant(1, tf.int32), tf.zeros([1, VOCAB]), lm.get_initial_state(1)),
        )[1]

    states = lm.get_initial_state(1)
    expected = np.zeros([1, VOCAB], dtype=np.float32)
    for token in (1, 2, 3):
        scores, states = lm.call_next(tf.constant([[token]], tf.int32), states)
        expected += scores.numpy()
    np.testing.assert_allclose(loop().numpy(), expected, rtol=1e-5)


def test_state_survives_the_beams_reordering():
    """
    The beam re-orders states on axis 0 when hypotheses recombine. State is [B, V] here, so this
    checks that a gathered row still decodes to the same thing it did before the shuffle.
    """
    lm, _ = fit()
    states = lm.get_initial_state(3)
    scores, states = lm.call_next(tf.constant([[1], [2], [3]], tf.int32), states)

    shuffled = tf.gather(states, tf.constant([2, 0, 1], tf.int32), axis=0)
    following = tf.constant([[2], [2], [2]], tf.int32)
    original, _ = lm.call_next(following, states)
    reordered, _ = lm.call_next(following, shuffled)
    np.testing.assert_allclose(reordered.numpy(), original.numpy()[[2, 0, 1]], rtol=1e-6)


# --------------------------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------------------------


def test_round_trips_through_config_and_h5(tmp_path):
    """How `train_lm` writes it and `build_lm` reads it back."""
    lm, _ = fit()
    path = str(tmp_path / "ngram.weights.h5")
    lm.save_weights(path)

    blob = keras.saving.serialize_keras_object(lm)
    assert blob["config"]["order"] == ORDER and blob["config"]["max_arcs"] == 4096
    assert "arc_keys" not in blob["config"], "the tables are weights, they must not be inlined in the config"

    restored = BaseModel.build_lm(blob, weights=path)
    np.testing.assert_array_equal(restored.arc_keys.numpy(), lm.arc_keys.numpy())
    np.testing.assert_allclose(restored.arc_weights.numpy(), lm.arc_weights.numpy())
    np.testing.assert_array_equal(restored.bos_state.numpy(), lm.bos_state.numpy())

    scores, _ = restored.call_next(tf.constant([[1]], tf.int32), restored.get_initial_state(1))
    original, _ = lm.call_next(tf.constant([[1]], tf.int32), lm.get_initial_state(1))
    np.testing.assert_allclose(scores.numpy(), original.numpy(), rtol=1e-6)


def test_an_unfitted_model_is_well_defined():
    """`build_lm` builds before it loads, so an unfitted model must not produce garbage."""
    lm = NGramLanguageModel(vocab_size=VOCAB, blank=BLANK, order=ORDER, max_arcs=64, max_states=16)
    scores, states = lm.call_next(tf.constant([[1]], tf.int32), lm.get_initial_state(1))
    assert np.all(scores.numpy() == 0.0), "every lookup must miss, not match the zero-initialised table"
    assert np.isfinite(scores.numpy()).all()
    assert states.shape == (1, VOCAB)


# --------------------------------------------------------------------------------------------
# the ARPA reader
# --------------------------------------------------------------------------------------------


ARPA = """\\data\\
ngram 1=5
ngram 2=4

\\1-grams:
-99\t<s>\t-0.30103
-1.30103\t</s>
-2.0\t<unk>
-0.60206\t1\t-0.15
-0.69897\t2\t-0.20
-1.00000\t3\t-0.10

\\2-grams:
-0.30103\t<s> 1
-0.39794\t1 2
-0.52288\t2 3
-0.60206\t1 3

\\end\\
"""


def write_arpa(tmp_path, text=ARPA, name="lm.arpa"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def arpa_model(tmp_path, **kwargs):
    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    arrays = read_arpa(write_arpa(tmp_path), vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512, **kwargs)
    lm = NGramLanguageModel(vocab_size=VOCAB, blank=BLANK, order=arrays["stats"]["order"], max_arcs=4096, max_states=512)
    lm._assign(arrays)  # pylint: disable=protected-access
    return lm, arrays


def test_arpa_reads_orders_and_counts(tmp_path):
    _, arrays = arpa_model(tmp_path)
    assert arrays["stats"]["order"] == 2 and arrays["stats"]["arpa_order"] == 2
    assert arrays["stats"]["source"] == "arpa"
    assert arrays["stats"]["skipped_lines"] == 0


def test_arpa_converts_log10_to_natural_log(tmp_path):
    """ARPA is base 10 and the tensors are natural log -- forgetting this scales every score by 2.3."""
    lm, _ = arpa_model(tmp_path)
    # p(2 | 1) is -0.39794 in log10 == 0.4 linear. The unigram level is renormalised, but a bigram
    # arc is taken from the file unchanged, so it must come back as ln(0.4).
    scores = reference_scores(lm, (1,))
    np.testing.assert_allclose(np.exp(scores[2]), 0.4, rtol=1e-4)


def test_arpa_is_a_distribution_over_labels(tmp_path):
    lm, _ = arpa_model(tmp_path)
    for context in [(), (1,), (2,), (5,)]:
        scores = reference_scores(lm, context)
        assert set(scores) == set(labels()), f"context {context} left a label unscored"
        assert np.isfinite(list(scores.values())).all()
        np.testing.assert_allclose(sum(np.exp(list(scores.values()))), 1.0, atol=1e-6, err_msg=f"context {context}")


def test_arpa_floors_unlisted_labels_with_unk(tmp_path):
    """Labels 4 and 5 never appear in the file; without the <unk> floor they would be ln 0."""
    lm, _ = arpa_model(tmp_path)
    unigram = reference_scores_from_state(lm, 0)
    assert np.isfinite(unigram[4]) and np.isfinite(unigram[5])
    # <unk> is -2.0 log10, rarer than every listed unigram, so the floored labels must rank last
    assert unigram[4] < unigram[1] and unigram[5] < unigram[3]
    np.testing.assert_allclose(unigram[4], unigram[5], rtol=1e-6), "both floored labels get the same mass"


def test_arpa_maps_bos_to_blank_and_drops_eos(tmp_path):
    """`<s>` is the start marker; `</s>` has no counterpart here and must not become a label."""
    lm, _ = arpa_model(tmp_path)
    # `<s> 1` is the strongest bigram in the file, so the start state must prefer 1 over the unigram
    start = reference_scores(lm, ())
    unigram = reference_scores_from_state(lm, 0)
    assert start[1] > unigram[1], "the <s> context must reach the start state"
    # and nothing anywhere may score blank
    scores, _ = lm.call_next(tf.constant([[1]], tf.int32), lm.get_initial_state(1))
    assert scores.numpy()[0, BLANK] == 0.0


def test_arpa_truncates_to_a_requested_lower_order(tmp_path):
    _, full = arpa_model(tmp_path)
    _, unigram_only = arpa_model(tmp_path, order=1)
    assert unigram_only["stats"]["order"] == 1
    assert unigram_only["stats"]["arcs"] < full["stats"]["arcs"], "dropping to order 1 must drop the bigram arcs"


def test_arpa_ignores_unmappable_words(tmp_path):
    """A word-level ARPA cannot be used; it must be reported, not silently half-read."""
    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    words = ARPA.replace("\t1\t", "\thello\t").replace("<s> 1", "<s> hello").replace("\t1 2", "\thello 2")
    path = write_arpa(tmp_path, words, name="words.arpa")
    arrays = read_arpa(path, vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512)
    assert arrays["stats"]["skipped_lines"] >= 2, "the unmappable entries must be counted as skipped"


def test_arpa_rejects_a_file_with_no_usable_unigrams(tmp_path):
    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    empty = "\\data\\\nngram 1=1\n\n\\1-grams:\n-1.0\t</s>\n\n\\end\\\n"
    with pytest.raises(ValueError, match="No usable unigrams"):
        read_arpa(write_arpa(tmp_path, empty, name="empty.arpa"), vocab_size=VOCAB, blank=BLANK)


def test_arpa_reads_gzip(tmp_path):
    import gzip as gziplib

    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    path = tmp_path / "lm.arpa.gz"
    with gziplib.open(path, "wt", encoding="utf-8") as handle:
        handle.write(ARPA)
    arrays = read_arpa(str(path), vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512)
    assert arrays["stats"]["arcs"] > 0


def test_arpa_model_decodes_through_call_next(tmp_path):
    """The end of the road: an ARPA-built model must drive the decoder like any other."""
    lm, _ = arpa_model(tmp_path)
    states = lm.get_initial_state(2)
    scores, states = lm.call_next(tf.constant([[1], [2]], tf.int32), states)
    assert scores.shape == (2, VOCAB) and np.isfinite(scores.numpy()).all()
    without_blank = np.delete(np.exp(scores.numpy().astype(np.float64)), BLANK, axis=-1)
    np.testing.assert_allclose(without_blank.sum(-1), 1.0, atol=1e-5)


# ARPA with </s> listed in every context, and a context that lists *every* label. Both are normal in
# a real lmplz file, and each broke normalisation in a different way before being handled.
ARPA_EOS = """\\data\\
ngram 1=7
ngram 2=7

\\1-grams:
-99\t<s>\t-0.30103
-0.69897\t</s>
-2.00000\t<unk>
-0.79588\t1\t-0.15
-0.79588\t2\t-0.20
-0.79588\t3\t-0.10
-0.79588\t4\t-0.10
-0.79588\t5\t-0.10

\\2-grams:
-0.39794\t<s> 1
-0.39794\t<s> </s>
-0.69897\t1 1
-0.69897\t1 2
-0.69897\t1 3
-0.69897\t1 4
-0.69897\t1 5

\\end\\
"""


def test_arpa_conditions_away_eos_rather_than_dropping_it(tmp_path):
    """
    `</s>` holds real mass in every context. Deleting those entries and leaving the rest alone would
    leave each state summing to `1 - p(</s>)`, a silently sub-normalised LM the beam cannot detect.
    """
    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    path = write_arpa(tmp_path, ARPA_EOS, name="eos.arpa")
    arrays = read_arpa(path, vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512)
    lm = NGramLanguageModel(vocab_size=VOCAB, blank=BLANK, order=2, max_arcs=4096, max_states=512)
    lm._assign(arrays)  # pylint: disable=protected-access

    # context `<s>` gives half its mass to </s>; the labels left must still be a distribution
    start = reference_scores(lm, ())
    np.testing.assert_allclose(sum(np.exp(list(start.values()))), 1.0, atol=1e-6)


def test_arpa_state_listing_every_label_is_still_normalised(tmp_path):
    """
    Context `1` lists all five labels, so backoff has nowhere to deposit the discount mass and
    adjusting the backoff weight cannot fix the total -- the arcs have to be rescaled instead.
    """
    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    path = write_arpa(tmp_path, ARPA_EOS, name="full.arpa")
    arrays = read_arpa(path, vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512)
    lm = NGramLanguageModel(vocab_size=VOCAB, blank=BLANK, order=2, max_arcs=4096, max_states=512)
    lm._assign(arrays)  # pylint: disable=protected-access

    scores = reference_scores(lm, (1,))
    assert set(scores) == set(labels())
    np.testing.assert_allclose(sum(np.exp(list(scores.values()))), 1.0, atol=1e-6)
    # they were equal in the file, so they must stay equal after rescaling
    values = [scores[w] for w in labels()]
    np.testing.assert_allclose(values, values[0], rtol=1e-6)


def test_arpa_rejects_a_context_carrying_more_than_all_the_probability(tmp_path):
    """
    The signature of a word-level ARPA read as token-level: many words collapse onto one id and
    their probabilities stack. It must fail loudly rather than decode with a broken LM.
    """
    from tensorflow_asr.models.lm.ngram_language_model import read_arpa

    broken = ARPA.replace("-0.39794\t1 2", "-0.09691\t1 2").replace("-0.60206\t1 3", "-0.09691\t1 3")
    broken = broken.replace("\\2-grams:", "\\2-grams:\n-0.09691\t1 4\n-0.09691\t1 5")
    path = write_arpa(tmp_path, broken, name="overfull.arpa")
    with pytest.raises(ValueError, match="more probability than one"):
        read_arpa(path, vocab_size=VOCAB, blank=BLANK, max_arcs=4096, max_states=512)
