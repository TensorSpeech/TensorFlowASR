"""
Correctness tests for the LODR bigram language model.

Two things have to hold for this to be usable as an internal LM estimate. It must be a proper
conditional distribution over the labels -- every row summing to one, no `-inf` anywhere, blank
inert -- because the beam adds these numbers into live hypotheses and a single `-inf` kills one
outright. And it must actually reflect the counts, not just be well-formed.
"""

import numpy as np
import pytest

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.bigram_language_model import BigramLanguageModel, build_table, count_bigrams

BLANK = 0
VOCAB = 6


def probs(table, blank=BLANK):
    """Label probabilities of each row, blank excluded."""
    return np.delete(np.exp(np.asarray(table, dtype=np.float64)), blank, axis=-1)


# --------------------------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------------------------


def test_counts_pairs_and_seeds_sentence_start():
    counts = count_bigrams([[1, 2, 3], [1, 2]], vocab_size=VOCAB, blank=BLANK)
    assert counts[BLANK, 1] == 2, "row `blank` must count sentence starts"
    assert counts[1, 2] == 2
    assert counts[2, 3] == 1
    assert counts.sum() == 5  # 2 starts + 3 transitions
    assert counts[3].sum() == 0, "the last token of a transcript starts no pair"


def test_counts_repeated_pairs():
    """`counts[ctx, tok] += 1` would collapse a repeat to one; `np.add.at` must not."""
    counts = count_bigrams([[1, 1, 1, 1]], vocab_size=VOCAB, blank=BLANK)
    assert counts[1, 1] == 3, f"expected 3 self-transitions, got {counts[1, 1]}"
    assert counts[BLANK, 1] == 1


def test_counts_ignore_out_of_vocabulary_and_empty():
    counts = count_bigrams([[1, VOCAB + 4, 2], [], [-1]], vocab_size=VOCAB, blank=BLANK)
    assert counts.sum() == 2  # the out-of-range token is dropped, leaving 1 then 2 as two starts
    assert counts.shape == (VOCAB, VOCAB)


# --------------------------------------------------------------------------------------------
# the table is a valid distribution
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("interpolation", [0.0, 0.5, 0.75, 0.999])
def test_every_row_is_a_distribution_over_labels(interpolation):
    counts = count_bigrams([[1, 2, 3], [3, 1], [2, 2, 4]], vocab_size=VOCAB, blank=BLANK)
    table = build_table(counts, blank=BLANK, interpolation=interpolation)

    assert np.isfinite(table).all(), "an -inf entry would kill a legal hypothesis in the beam"
    np.testing.assert_allclose(probs(table).sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(table[:, BLANK] == 0.0), "blank must be inert under subtraction, ie. exactly 0"


def test_unseen_context_falls_back_to_the_unigram():
    """A row with no counts must still sum to one, not to 1 - interpolation."""
    counts = count_bigrams([[1, 2]], vocab_size=VOCAB, blank=BLANK)
    table = build_table(counts, blank=BLANK, interpolation=0.75)

    unseen = 4  # never appeared as a context
    assert counts[unseen].sum() == 0
    np.testing.assert_allclose(probs(table)[unseen].sum(), 1.0, atol=1e-6)

    # and it must equal the unigram exactly, since there is nothing else to interpolate with
    same = 5
    np.testing.assert_allclose(table[unseen], table[same], rtol=1e-6)


def test_counts_move_probability_mass():
    """Guards against a table that is well-formed but ignores the data."""
    counts = count_bigrams([[1, 2]] * 50, vocab_size=VOCAB, blank=BLANK)
    table = build_table(counts, blank=BLANK, interpolation=0.75)
    assert table[1, 2] > table[1, 3], "the observed successor must outrank an unobserved one"
    assert table[BLANK, 1] > table[BLANK, 4], "sentence-start counts must reach row blank"


def test_interpolation_zero_ignores_the_counts():
    counts = count_bigrams([[1, 2]] * 50, vocab_size=VOCAB, blank=BLANK)
    table = build_table(counts, blank=BLANK, interpolation=0.0)
    assert np.allclose(table, table[0][None, :]), "at lam=0 every row must be the same unigram"


def test_rejects_bad_smoothing():
    counts = count_bigrams([[1, 2]], vocab_size=VOCAB, blank=BLANK)
    with pytest.raises(ValueError, match="delta"):
        build_table(counts, blank=BLANK, delta=0.0)
    with pytest.raises(ValueError, match="interpolation"):
        build_table(counts, blank=BLANK, interpolation=1.5)
    # 1.0 is the interesting one: it looks like a legal "trust the counts fully" setting, but it
    # leaves a seen context with no unigram floor and sends unseen successors to -inf
    with pytest.raises(ValueError, match="interpolation"):
        build_table(counts, blank=BLANK, interpolation=1.0)


def test_matches_ilme_output_shape_conventions():
    """
    LODR and ILME are subtracted by the same code path, so their tables must agree in form:
    blank exactly 0, labels normalised on their own. Otherwise `lm_beta` would mean two things.
    """
    from tensorflow_asr.models.transducer.base_transducer import _internal_lm_log_probs

    counts = count_bigrams([[1, 2, 3], [2, 4]], vocab_size=VOCAB, blank=BLANK)
    lodr = build_table(counts, blank=BLANK)
    ilme = _internal_lm_log_probs(tf.random.stateless_normal([4, VOCAB], seed=[1, 2]), BLANK, VOCAB).numpy()

    assert np.all(lodr[:, BLANK] == 0.0) and np.all(ilme[:, BLANK] == 0.0)
    np.testing.assert_allclose(probs(lodr).sum(-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(probs(ilme).sum(-1), 1.0, atol=1e-5)




# --------------------------------------------------------------------------------------------
# the keras model
# --------------------------------------------------------------------------------------------


@pytest.fixture
def fitted():
    """A bigram fitted by counting, the way `train_lm --target=internal` produces one."""
    lm = BigramLanguageModel(vocab_size=VOCAB, blank=BLANK)
    lm.fit_counts([[1, 2, 3], [3, 1], [2, 2, 4]])
    return lm


def test_fit_counts_fills_the_weight(fitted):
    """`fit_counts` replaces gradient training, so it has to actually move the variable."""
    fresh = BigramLanguageModel(vocab_size=VOCAB, blank=BLANK)
    assert np.all(fresh.table.numpy() == 0.0), "a fresh model starts empty"
    assert not np.all(fitted.table.numpy() == 0.0)

    counts = count_bigrams([[1, 2, 3], [3, 1], [2, 2, 4]], vocab_size=VOCAB, blank=BLANK)
    np.testing.assert_allclose(fitted.table.numpy(), build_table(counts, blank=BLANK), rtol=1e-6)
    assert fitted.table.trainable is False, "counting is the estimate; no gradient should move it"


def test_call_next_reads_the_row_of_the_previous_token(fitted):
    """The decoding path: one step, state passed straight through."""
    scores, states = fitted.call_next(tf.constant([[1], [3], [BLANK]], tf.int32), fitted.get_initial_state(3))

    assert scores.shape == (3, VOCAB)
    np.testing.assert_allclose(scores.numpy(), fitted.table.numpy()[[1, 3, BLANK]], rtol=1e-6)
    # a bigram carries nothing, so the state must come back untouched for the beam to re-order
    np.testing.assert_array_equal(states.numpy(), fitted.get_initial_state(3).numpy())


def test_call_scores_a_whole_sequence(fitted):
    """The training path: [B, U] -> [B, U, V], and it must agree with the decoding path."""
    tokens = tf.constant([[BLANK, 1, 2], [BLANK, 3, 1]], tf.int32)
    outputs = fitted(tokens)  # through __call__, the way `fit` reaches it

    assert outputs.shape == (2, 3, VOCAB)
    for u in range(3):
        step, _ = fitted.call_next(tokens[:, u : u + 1], fitted.get_initial_state(2))
        np.testing.assert_allclose(outputs.numpy()[:, u], step.numpy(), rtol=1e-6)


def test_call_does_not_autocast_the_token_indices(fitted):
    """
    Keras autocasts floating-point inputs to the compute dtype. Token indices are int32 and must
    come through untouched -- a float index would break the `tf.gather` outright.
    """
    seen = {}
    original = fitted.call

    def spy(tokens, training=False):
        seen["dtype"] = tokens.dtype
        return original(tokens, training=training)

    fitted.call = spy
    fitted(tf.constant([[2, 3]], tf.int32))
    assert seen["dtype"] == tf.int32, f"keras cast the indices to {seen['dtype']}"


def test_call_next_works_inside_a_while_loop(fitted):
    """
    The real constraint: the decoder calls this from inside a `tf.while_loop` body. A model that
    built lazily would be creating variables there, which graph mode rejects.
    """

    @tf.function
    def loop():
        def body(i, acc, states):
            scores, states = fitted.call_next(tf.fill([1, 1], i), states)
            return i + 1, acc + scores, states

        return tf.while_loop(
            lambda i, *_: i < 4,  # i = 1, 2, 3
            body,
            (tf.constant(1, tf.int32), tf.zeros([1, VOCAB]), fitted.get_initial_state(1)),
        )[1]

    table = fitted.table.numpy()
    np.testing.assert_allclose(loop().numpy()[0], table[1] + table[2] + table[3], rtol=1e-5)


def test_round_trips_through_config_and_h5(fitted, tmp_path):
    """How `train_lm` writes it and `build_lm` reads it back."""
    path = str(tmp_path / "bigram.weights.h5")
    fitted.save_weights(path)

    blob = keras.saving.serialize_keras_object(fitted)
    assert blob["config"]["vocab_size"] == VOCAB and blob["config"]["blank"] == BLANK
    assert "table" not in blob["config"], "the table is a weight, it must not be inlined in the config"

    restored = BaseModel.build_lm(blob, weights=path)
    np.testing.assert_allclose(restored.table.numpy(), fitted.table.numpy())

    # and without weights it is empty, ie. the h5 is doing the work rather than the config
    assert np.all(BaseModel.build_lm(blob).table.numpy() == 0.0)
