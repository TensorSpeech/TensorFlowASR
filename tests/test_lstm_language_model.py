"""
Correctness tests for the LSTM language model fused into beam search.

The property everything else rests on is that the two forward passes compute the same function:
`call` teacher-forced over a whole sequence, and `call_next` stepped one token at a time carrying
state. Training uses the first, decoding uses the second, so a mismatch would train one model and
decode with another -- silently, since both paths look fine in isolation.
"""

import numpy as np
import pytest

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.lstm_language_model import LSTMLanguageModel

VOCAB = 12
BLANK = 0


def build(**kwargs):
    lm = LSTMLanguageModel(**{"vocab_size": VOCAB, "embed_dim": 8, "units": 16, "nlayers": 2, **kwargs})
    lm.make()
    return lm


@pytest.fixture
def lm():
    return build()


# --------------------------------------------------------------------------------------------
# the two forward passes must agree
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nlayers", [1, 2, 3])
@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_call_next_threaded_matches_call_teacher_forced(nlayers, tie_embeddings):
    lm = build(nlayers=nlayers, tie_embeddings=tie_embeddings)
    tokens = tf.constant([[BLANK, 3, 5, 2, 9], [BLANK, 7, 1, 4, 6]], tf.int32)

    sequence = lm(tokens).numpy()  # [B, U, V], the training path
    states, stepped = lm.get_initial_state(2), []
    for u in range(tokens.shape[1]):
        output, states = lm.call_next(tokens[:, u : u + 1], states)
        stepped.append(output.numpy())

    np.testing.assert_allclose(np.stack(stepped, axis=1), sequence, rtol=1e-5, atol=1e-5)


def test_outputs_are_log_probabilities(lm):
    tokens = tf.constant([[BLANK, 3, 5]], tf.int32)
    np.testing.assert_allclose(np.exp(lm(tokens).numpy()).sum(-1), 1.0, rtol=1e-5)
    step, _ = lm.call_next(tf.constant([[3]], tf.int32), lm.get_initial_state(1))
    np.testing.assert_allclose(np.exp(step.numpy()).sum(-1), 1.0, rtol=1e-5)


def test_state_layout_matches_the_beam_contract(lm):
    """One tensor, batch on axis 0 -- the beam re-orders it on that axis during recombination."""
    state = lm.get_initial_state(5)
    assert state.shape == (5, 2, 2, 16), "[B, nlayers, 2 (h and c), units]"
    assert np.all(state.numpy() == 0.0)

    _, updated = lm.call_next(tf.constant([[3]] * 5, tf.int32), state)
    assert updated.shape == state.shape
    assert not np.all(updated.numpy() == 0.0), "the state must actually advance"


def test_state_is_what_carries_the_history(lm):
    """Same token, different state, must give a different distribution -- otherwise it is a unigram."""
    token = tf.constant([[4]], tf.int32)
    first, state = lm.call_next(token, lm.get_initial_state(1))
    second, _ = lm.call_next(token, state)
    assert not np.allclose(first.numpy(), second.numpy(), atol=1e-6)


# --------------------------------------------------------------------------------------------
# where it gets used
# --------------------------------------------------------------------------------------------


def test_call_next_works_inside_a_while_loop(lm):
    """The decoder calls this from inside a `tf.while_loop`, where creating variables would fail."""

    @tf.function
    def loop():
        def body(i, states, total):
            output, states = lm.call_next(tf.fill([3, 1], i), states)
            return i + 1, states, total + tf.reduce_sum(output)

        return tf.while_loop(lambda i, *_: i < 4, body, (tf.constant(1, tf.int32), lm.get_initial_state(3), tf.constant(0.0)))[2]

    assert np.isfinite(float(loop()))


def test_graph_matches_eager(lm):
    """`call_next` is traced into the beam's graph; it must compute the same thing there."""
    tokens, state = tf.constant([[3], [7]], tf.int32), lm.get_initial_state(2)
    eager_out, eager_state = lm.call_next(tokens, state)
    graph_out, graph_state = tf.function(lm.call_next)(tokens, state)
    np.testing.assert_array_equal(eager_out.numpy(), graph_out.numpy())
    np.testing.assert_array_equal(eager_state.numpy(), graph_state.numpy())


def test_round_trips_through_config_and_h5(lm, tmp_path):
    path = str(tmp_path / "lm.weights.h5")
    lm.save_weights(path)

    blob = keras.saving.serialize_keras_object(lm)
    assert blob["config"]["units"] == 16 and blob["config"]["nlayers"] == 2
    restored = BaseModel.build_lm(blob, weights=path)

    tokens, state = tf.constant([[5], [2]], tf.int32), lm.get_initial_state(2)
    np.testing.assert_allclose(restored.call_next(tokens, state)[0].numpy(), lm.call_next(tokens, state)[0].numpy(), rtol=1e-6)


# --------------------------------------------------------------------------------------------
# the published architecture
# --------------------------------------------------------------------------------------------


def test_tied_embeddings_are_smaller_and_reuse_the_matrix():
    tied, untied = build(tie_embeddings=True), build(tie_embeddings=False)
    assert tied.count_params() != untied.count_params()
    # tying means there is no separate output kernel; the projection to embed_dim replaces it
    assert tied.output_dense is None and tied.projection is not None
    assert untied.output_dense is not None and untied.projection is None


def test_ilme_defaults_reproduce_the_published_size():
    """
    The ILME paper's external LM: 2 x 2048 LSTM, 512-dim tied embeddings, 3999 word-pieces, 58M
    parameters (https://arxiv.org/abs/2011.01991). The defaults here should land on that.
    """
    lm = LSTMLanguageModel(vocab_size=3999)
    lm.make()
    millions = lm.count_params() / 1e6
    assert 55 < millions < 61, f"expected ~58M parameters, got {millions:.1f}M"
