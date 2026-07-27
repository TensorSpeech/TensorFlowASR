# pylint: disable=redefined-outer-name
"""
Correctness tests for the ALSD++ transducer beam search (`Transducer.recognize_beam`).

The transducer is driven by a stubbed joint network whose log-probabilities depend only on
`(frame, previous token)`, and the encoder output carries the frame and batch index in its first
two channels. That makes the whole search space small enough to solve exactly with a dynamic
program over the `(t, u, last, expansions)` lattice, so the beam can be compared against a known
optimum instead of merely against itself.

Two regimes matter and the distinction drives most of the parametrization below:

* when `max_tokens_per_frame` never binds and the beam is wide enough to be exhaustive, ALSD++
  **must** reproduce the DP optimum -- any mismatch is a bug;
* when the cap binds or the beam is narrow, pruning and transcript-hash recombination are both
  lossy by design, so only legality can be asserted.
"""

import logging

import numpy as np
import pytest

from tensorflow_asr import keras, schemas, tf
from tensorflow_asr.configs import DecoderConfig, LanguageModelConfig
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.models.lm.language_model import LanguageModel
from tensorflow_asr.models.transducer.base_transducer import _internal_lm_log_probs
from tensorflow_asr.models.transducer.rnnt import RnnTransducer
from tensorflow_asr.utils import app_util

BLANK = 0

SPEECH_CONFIG = {
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "nfft": 512,
    "feature_type": "log_mel_spectrogram",
}


LM_POSITIONS = 3  # distinct conditioning buckets of the test LM


@keras.utils.register_keras_serializable(package="tests.test_beam_search")
class CountingLanguageModel(LanguageModel):
    """
    Bigram LM whose distribution also depends on how many labels the hypothesis has emitted.

    The state is that count. `call_next` runs *before* the next token is chosen, so an LM cannot
    condition on it -- conditioning comes from `previous_tokens`, and the state carries the
    recurrent part, exactly as the prediction network does. Making the distribution depend on the
    count means a beam that advanced the state on a blank, or failed to re-order states onto the
    selected parents, produces different scores and fails the comparison against the DP.
    """

    def __init__(self, table, **kwargs):
        super().__init__(**kwargs)
        table = np.asarray(table, dtype=np.float32)  # [LM_POSITIONS, V, V]
        # A weight, not a constant, so this saves and loads through the ordinary keras h5 path the
        # way a real LM does. Created here rather than in `build` because the decoder calls
        # `call_next` from inside a `tf.while_loop`, where creating variables fails.
        self.table = self.add_weight(name="table", shape=table.shape, initializer="zeros", trainable=False, dtype="float32")
        self.table.assign(table)

    def get_initial_state(self, batch_size):
        return tf.zeros([batch_size, 1], dtype=tf.int32)

    def call(self, tokens, training=False):
        # Training path: position u is scored from the bucket it sits in, which is what the state
        # counts up to at decoding time.
        positions = tf.minimum(tf.range(tf.shape(tokens)[1]), LM_POSITIONS - 1)  # [U]
        positions = tf.tile(tf.expand_dims(positions, 0), [tf.shape(tokens)[0], 1])  # [B, U]
        return tf.gather_nd(self.table, tf.stack([positions, tokens], axis=-1))  # [B, U, V]

    def call_next(self, previous_tokens, previous_states):
        count = tf.minimum(tf.reshape(previous_states, [-1]), LM_POSITIONS - 1)
        indices = tf.stack([count, tf.reshape(previous_tokens, [-1])], axis=-1)
        return tf.gather_nd(self.table, indices), tf.reshape(previous_states, [-1, 1]) + 1

    def compute_output_shape(self, tokens_shape):
        return (*tokens_shape, self.table.shape[-1])

    def get_config(self):
        # `build_lm` deserializes from a keras blob, so the tests that go through it need this
        return {**super().get_config(), "table": self.table.numpy().tolist()}


def log_softmax(seed, shape, scale=2.0):
    logits = np.random.RandomState(seed).randn(*shape).astype(np.float32) * scale
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def internal_log_softmax(seed, shape, scale=2.0):
    """
    An internal LM distribution in the form `_internal_lm_log_probs` produces: normalised over the
    labels alone, with 0.0 in the blank column so that subtracting it never touches blank.
    """
    logits = np.random.RandomState(seed).randn(*shape).astype(np.float32) * scale
    logits[..., BLANK] = -np.inf
    out = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    out[..., BLANK] = 0.0
    return out


def token_budget(nframes, max_tokens_per_frame, padded_frames=None):
    """`u` cap of the implementation: max_frames * 2 + 1, on the *padded* batch width."""
    return min(nframes * max_tokens_per_frame, 2 * (padded_frames or nframes) + 1)


def solve_exact(
    logp,
    nframes,
    vocab_size,
    max_tokens_per_frame,
    score_norm,
    lm_logp=None,
    lm_alpha=0.0,
    ilm_logp=None,
    lm_beta=0.0,
    padded_frames=None,
):
    """
    Forward DP over `(t, u, last, expansions)` returning the best legal label sequence.

    Transitions are scored with the plain transducer log-probabilities when neither LM is given,
    and otherwise with eq. (3) of https://arxiv.org/abs/2506.00185 plus the internal LM subtraction
    of eq. (27) of https://arxiv.org/abs/2011.01991.

    `ilm_logp` is read as `[state, last, token]` when it is 3-D -- the stateful LODR case, sharing
    the test LM's convention that the state is the label count -- and as `[last, token]` when it is
    2-D, the ILME case where the internal LM rides on the prediction network's own state.
    """
    max_u = token_budget(nframes, max_tokens_per_frame, padded_frames)
    fused = lm_logp is not None or ilm_logp is not None

    def cost(t, u, last, token):
        # `u` doubles as the test LM's state: the number of labels emitted so far
        if token == BLANK:
            return (1.0 + lm_alpha) * logp[t, last, BLANK] if fused else logp[t, last, BLANK]
        if not fused:
            return logp[t, last, token]
        # The external LM contributes 0 when absent, which is what the implementation does too:
        # the `lm_alpha * ln(1 - p[blank])` half of eq. (3) survives on its own.
        lm_value = 0.0 if lm_logp is None else lm_logp[min(u, LM_POSITIONS - 1), last, token]
        log_not_blank = np.log(-np.expm1(min(logp[t, last, BLANK], -1e-7)))
        value = logp[t, last, token] + lm_alpha * (log_not_blank + lm_value)
        if ilm_logp is not None:
            ilm_value = ilm_logp[min(u, LM_POSITIONS - 1), last, token] if ilm_logp.ndim == 3 else ilm_logp[last, token]
            value -= lm_beta * ilm_value
        return value

    states = {(0, 0, BLANK, 0): (0.0, ())}
    terminals = []
    for alignment_length in range(nframes + max_u + 1):
        current = [(k, v) for k, v in states.items() if k[0] + k[1] == alignment_length]
        for (t, u, last, expansions), (score, seq) in current:
            if t == nframes:
                terminals.append((score, u, seq))
                continue
            moves = [((t + 1, u, last, 0), score + cost(t, u, last, BLANK), seq)]
            if expansions < max_tokens_per_frame and u < max_u:
                for token in range(1, vocab_size):
                    moves.append(((t, u + 1, token, expansions + 1), score + cost(t, u, last, token), seq + (token,)))
            for key, new_score, new_seq in moves:
                if key not in states or new_score > states[key][0]:
                    states[key] = (new_score, new_seq)

    rank = (lambda x: x[0] / max(x[1], 1)) if score_norm else (lambda x: x[0])
    return list(max(terminals, key=rank)[2])


def build_model(vocab_size):
    model = RnnTransducer(
        blank=BLANK,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_reduction_positions=["post"],
        encoder_reduction_factors=[1],
        encoder_dmodel=8,
        encoder_nlayers=1,
        encoder_rnn_units=8,
        prediction_embed_dim=8,
        prediction_num_rnns=1,
        prediction_rnn_units=8,
        prediction_projection_units=8,
        joint_dim=8,
    )
    model.make(batch_size=1)
    return model


def stub_transducer(model, logp, lengths, vocab_size, padded_frames, ilm_logp=None):
    """
    Replace feature extraction, encoder and joint with a table lookup on (batch, frame, last).

    `ilm_logp` is the [V, V] internal LM table the stub hands back when the beam asks for one, in
    place of re-running the joint with a zeroed encoder output. It is indexed by the last token
    only, since the internal LM of a transducer sees the label history and nothing else -- which is
    exactly the property `test_internal_lm_ignores_the_encoder` checks on a real model.
    """
    batch_size = len(lengths)
    encoded = np.zeros((batch_size, padded_frames, 4), dtype=np.float32)
    for b in range(batch_size):
        encoded[b, :, 0] = np.arange(padded_frames)  # frame index
        encoded[b, :, 1] = b  # batch index
    encoded = tf.convert_to_tensor(encoded)
    encoded_length = tf.convert_to_tensor(np.asarray(lengths, np.int32))
    table = tf.convert_to_tensor(logp)

    model.feature_extraction = lambda inputs, training=False: (encoded, encoded_length)
    model.encoder.call_next = lambda features, features_length, previous: (encoded, encoded_length, None)

    ilm_table = tf.convert_to_tensor(ilm_logp) if ilm_logp is not None else None

    def call_next(current_frames, previous_tokens, previous_decoder_states, return_internal_lm=False):
        frame = tf.cast(tf.round(current_frames[:, 0, 0]), tf.int32)
        batch = tf.cast(tf.round(current_frames[:, 0, 1]), tf.int32)
        indices = tf.stack([batch, frame, tf.reshape(previous_tokens, [-1])], axis=-1)
        outputs = tf.reshape(tf.gather_nd(table, indices), [-1, 1, 1, vocab_size])
        if not return_internal_lm:
            return outputs, previous_decoder_states
        ilm = tf.gather(ilm_table, tf.reshape(previous_tokens, [-1]))  # [B * W, V]
        return outputs, tf.reshape(ilm, [-1, 1, 1, vocab_size]), previous_decoder_states

    model.call_next = call_next


def make_inputs(batch_size):
    return schemas.PredictInput(
        inputs=tf.zeros([batch_size, 16000]),
        inputs_length=tf.constant([16000] * batch_size, tf.int32),
        previous_tokens=tf.ones([batch_size, 1], tf.int32) * BLANK,
        previous_encoder_states=None,
        previous_decoder_states=tf.zeros([batch_size, 1, 1, 1]),
    )


def decode(model, logp, lengths, vocab_size, padded_frames, ilm_logp=None, **kwargs):
    stub_transducer(model, logp, lengths, vocab_size, padded_frames, ilm_logp=ilm_logp)
    outputs = model.recognize_beam(inputs=make_inputs(len(lengths)), **kwargs)
    return [[int(t) for t in row if int(t) != BLANK] for row in outputs.tokens.numpy()]


# --------------------------------------------------------------------------------------------
# exactness: `s` never binds and the beam is exhaustive, so the DP optimum must be reproduced
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nframes,vocab_size,max_tokens_per_frame,beam_width", [(3, 3, 7, 256), (4, 3, 9, 512), (3, 4, 7, 512)])
@pytest.mark.parametrize("score_norm", [False, True])
def test_matches_exact_dp(nframes, vocab_size, max_tokens_per_frame, beam_width, score_norm):
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + nframes * 131 + vocab_size * 17, (nframes, vocab_size, vocab_size))
        expected = solve_exact(logp, nframes, vocab_size, max_tokens_per_frame, score_norm)
        actual = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            beam_width=beam_width,
            max_tokens_per_frame=max_tokens_per_frame,
            score_norm=score_norm,
        )[0]
        assert actual == expected, f"seed {seed}: got {actual}, DP optimum is {expected}"


@pytest.mark.parametrize("score_norm", [False, True])
def test_ragged_batch_matches_exact_dp(score_norm):
    """Each utterance must be decoded correctly even though the loop bound is batch-global."""
    nframes, vocab_size, max_tokens_per_frame, beam_width = 4, 3, 9, 512
    lengths = [4, 3, 1]
    model = build_model(vocab_size)
    for seed in range(3):
        logp = log_softmax(seed + 7000, (len(lengths), nframes, vocab_size, vocab_size))
        actual = decode(
            model,
            logp,
            lengths,
            vocab_size,
            nframes,
            beam_width=beam_width,
            max_tokens_per_frame=max_tokens_per_frame,
            score_norm=score_norm,
        )
        for b, length in enumerate(lengths):
            expected = solve_exact(logp[b], length, vocab_size, max_tokens_per_frame, score_norm, padded_frames=nframes)
            assert actual[b] == expected, f"seed {seed} utterance {b} (length {length}): got {actual[b]}, want {expected}"


# --------------------------------------------------------------------------------------------
# narrow beams: only legality is guaranteed, but widening must converge on the optimum
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("beam_width", [1, 2, 4, 16])
def test_narrow_beam_stays_legal(beam_width):
    nframes, vocab_size, max_tokens_per_frame = 6, 4, 2
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + 999, (nframes, vocab_size, vocab_size))
        actual = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            beam_width=beam_width,
            max_tokens_per_frame=max_tokens_per_frame,
        )[0]
        assert len(actual) <= token_budget(nframes, max_tokens_per_frame)
        assert all(1 <= token < vocab_size for token in actual)


def test_wider_beam_reaches_the_optimum():
    """A beam wide enough to be exhaustive finds the optimum that a width-1 beam misses."""
    nframes, vocab_size, max_tokens_per_frame = 6, 4, 2
    model = build_model(vocab_size)
    improved = 0
    for seed in range(6):
        logp = log_softmax(seed + 555, (nframes, vocab_size, vocab_size))
        expected = solve_exact(logp, nframes, vocab_size, max_tokens_per_frame, False)
        results = {
            beam_width: decode(
                model,
                logp[None],
                [nframes],
                vocab_size,
                nframes,
                beam_width=beam_width,
                max_tokens_per_frame=max_tokens_per_frame,
                score_norm=False,
            )[0]
            for beam_width in (1, 256)
        }
        assert results[256] == expected, f"seed {seed}: exhaustive beam missed the optimum"
        improved += results[1] != results[256]
    assert improved > 0, "width-1 and exhaustive beams never differed; the test is not exercising pruning"


# --------------------------------------------------------------------------------------------
# per-frame expansion cap
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_tokens_per_frame", [1, 2, 3])
def test_expansion_cap_is_respected(max_tokens_per_frame):
    """`s` bounds emissions per frame, which bounds the transcript by nframes * s."""
    nframes, vocab_size = 5, 4
    model = build_model(vocab_size)
    for seed in range(4):
        # peaked distributions that strongly prefer labels, so the cap is what stops emission
        logp = log_softmax(seed + 1234, (nframes, vocab_size, vocab_size), scale=6.0)
        actual = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            beam_width=8,
            max_tokens_per_frame=max_tokens_per_frame,
        )[0]
        assert len(actual) <= token_budget(nframes, max_tokens_per_frame)


# --------------------------------------------------------------------------------------------
# graph mode / determinism / output contract
# --------------------------------------------------------------------------------------------


def test_graph_mode_matches_eager_and_is_deterministic():
    nframes, vocab_size, max_tokens_per_frame, beam_width = 5, 4, 2, 8
    model = build_model(vocab_size)
    logp = log_softmax(4242, (nframes, vocab_size, vocab_size))
    stub_transducer(model, logp[None], [nframes], vocab_size, nframes)

    def run():
        return model.recognize_beam(inputs=make_inputs(1), beam_width=beam_width, max_tokens_per_frame=max_tokens_per_frame)

    eager = run()
    assert np.array_equal(run().tokens.numpy(), eager.tokens.numpy()), "decoding is not deterministic"
    graph = tf.function(run)()
    assert np.array_equal(graph.tokens.numpy(), eager.tokens.numpy()), "graph mode disagrees with eager"


def test_output_contract():
    """Shapes and streaming continuation fields must match what predict_step / TFLite expect."""
    nframes, vocab_size, batch_size = 5, 4, 3
    model = build_model(vocab_size)
    logp = log_softmax(31337, (batch_size, nframes, vocab_size, vocab_size))
    stub_transducer(model, logp, [nframes] * batch_size, vocab_size, nframes)

    outputs = model.recognize_beam(inputs=make_inputs(batch_size), beam_width=4)
    assert outputs.tokens.shape == (batch_size, nframes * 2 + 1)
    assert outputs.next_tokens.shape == (batch_size, 1)
    assert outputs.next_decoder_states.shape == (batch_size, 1, 1, 1)
    assert not np.isnan(outputs.next_decoder_states.numpy()).any()


# --------------------------------------------------------------------------------------------
# shallow fusion, eq. (3)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lm_alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("score_norm", [False, True])
def test_shallow_fusion_matches_exact_dp(lm_alpha, score_norm):
    nframes, vocab_size, max_tokens_per_frame, beam_width = 3, 3, 7, 256
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + 611, (nframes, vocab_size, vocab_size))
        lm_logp = log_softmax(seed + 4242, (LM_POSITIONS, vocab_size, vocab_size))
        expected = solve_exact(logp, nframes, vocab_size, max_tokens_per_frame, score_norm, lm_logp=lm_logp, lm_alpha=lm_alpha)
        actual = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            beam_width=beam_width,
            max_tokens_per_frame=max_tokens_per_frame,
            score_norm=score_norm,
            lm=CountingLanguageModel(lm_logp),
            lm_alpha=lm_alpha,
        )[0]
        assert actual == expected, f"seed {seed}: got {actual}, DP optimum is {expected}"


def test_zero_alpha_is_a_no_op():
    """A fused LM at alpha=0 must decode identically to no LM at all."""
    nframes, vocab_size = 4, 4
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + 313, (nframes, vocab_size, vocab_size))
        common = dict(beam_width=16, max_tokens_per_frame=2)
        without = decode(model, logp[None], [nframes], vocab_size, nframes, **common)[0]
        with_lm = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            lm=CountingLanguageModel(log_softmax(seed + 99, (LM_POSITIONS, vocab_size, vocab_size))),
            lm_alpha=0.0,
            **common,
        )[0]
        assert without == with_lm, f"seed {seed}: alpha=0 changed the hypothesis"


def test_language_model_changes_the_hypothesis():
    """Guards against fusion silently not reaching the scores at all."""
    nframes, vocab_size = 4, 4
    model = build_model(vocab_size)
    changed = 0
    for seed in range(8):
        logp = log_softmax(seed + 777, (nframes, vocab_size, vocab_size))
        common = dict(beam_width=16, max_tokens_per_frame=2)
        baseline = decode(model, logp[None], [nframes], vocab_size, nframes, **common)[0]
        fused = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            lm=CountingLanguageModel(log_softmax(seed + 555, (LM_POSITIONS, vocab_size, vocab_size), scale=4.0)),
            lm_alpha=1.5,
            **common,
        )[0]
        changed += baseline != fused
    assert changed > 0, "shallow fusion never altered the output"


def test_shallow_fusion_on_ragged_batch():
    nframes, vocab_size, max_tokens_per_frame, beam_width = 4, 3, 9, 512
    lengths = [4, 3, 1]
    model = build_model(vocab_size)
    logp = log_softmax(6000, (len(lengths), nframes, vocab_size, vocab_size))
    lm_logp = log_softmax(8000, (LM_POSITIONS, vocab_size, vocab_size))
    actual = decode(
        model,
        logp,
        lengths,
        vocab_size,
        nframes,
        beam_width=beam_width,
        max_tokens_per_frame=max_tokens_per_frame,
        score_norm=False,
        lm=CountingLanguageModel(lm_logp),
        lm_alpha=0.7,
    )
    for b, length in enumerate(lengths):
        expected = solve_exact(logp[b], length, vocab_size, max_tokens_per_frame, False, lm_logp=lm_logp, lm_alpha=0.7, padded_frames=nframes)
        assert actual[b] == expected, f"utterance {b} (length {length}): got {actual[b]}, want {expected}"


# --------------------------------------------------------------------------------------------
# internal LM subtraction: ILME (https://arxiv.org/abs/2011.01991), LODR (https://arxiv.org/abs/2203.16776)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [3, 5])
def test_internal_lm_log_probs_is_a_distribution_over_labels(vocab_size):
    """Blank must be inert under subtraction, and the labels must normalise on their own."""
    logits = tf.convert_to_tensor(np.random.RandomState(4).randn(2, 6, vocab_size).astype(np.float32) * 3.0)
    out = _internal_lm_log_probs(logits, BLANK, vocab_size).numpy()

    assert np.allclose(out[..., BLANK], 0.0), "blank column must be exactly 0, not a probability"
    labels = np.delete(out, BLANK, axis=-1)
    assert np.allclose(np.exp(labels).sum(axis=-1), 1.0, atol=1e-5), "labels must sum to one without blank"
    # the ordering of the label logits has to survive the renormalisation untouched
    kept = np.delete(logits.numpy(), BLANK, axis=-1)
    assert np.array_equal(np.argsort(labels, axis=-1), np.argsort(kept, axis=-1))


def test_internal_lm_ignores_the_encoder():
    """
    The defining property of the internal LM: it is a function of the label history alone.

    This runs the real joint network -- no stub -- so it is what actually pins down that zeroing
    the encoder output is what isolates the internal LM. Two very different acoustic frames must
    give the same internal LM score for the same previous token, while the ordinary output must
    not.
    """
    model = build_model(vocab_size=6)
    previous_tokens = tf.constant([[1], [2], [3]], tf.int32)
    states = model.predict_net.get_initial_state(batch_size=3)
    frames_a = tf.random.stateless_normal([3, 1, 8], seed=[1, 2])
    frames_b = tf.random.stateless_normal([3, 1, 8], seed=[3, 4]) * 5.0

    outputs_a, ilm_a, _ = model.call_next(frames_a, previous_tokens, states, return_internal_lm=True)
    outputs_b, ilm_b, _ = model.call_next(frames_b, previous_tokens, states, return_internal_lm=True)

    np.testing.assert_allclose(ilm_a.numpy(), ilm_b.numpy(), rtol=1e-5, atol=1e-6)
    assert not np.allclose(outputs_a.numpy(), outputs_b.numpy()), "the acoustic path is not reaching the output at all"
    assert np.allclose(ilm_a.numpy()[..., BLANK], 0.0)


@pytest.mark.parametrize("lm_beta", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("with_external_lm", [False, True])
def test_ilme_matches_exact_dp(lm_beta, with_external_lm):
    nframes, vocab_size, max_tokens_per_frame, beam_width = 3, 3, 7, 256
    lm_alpha = 0.6 if with_external_lm else 0.0
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + 1201, (nframes, vocab_size, vocab_size))
        ilm_logp = internal_log_softmax(seed + 1301, (vocab_size, vocab_size))
        lm_logp = log_softmax(seed + 1401, (LM_POSITIONS, vocab_size, vocab_size)) if with_external_lm else None
        expected = solve_exact(
            logp,
            nframes,
            vocab_size,
            max_tokens_per_frame,
            score_norm=False,
            lm_logp=lm_logp,
            lm_alpha=lm_alpha,
            ilm_logp=ilm_logp,
            lm_beta=lm_beta,
        )
        actual = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            ilm_logp=ilm_logp,
            beam_width=beam_width,
            max_tokens_per_frame=max_tokens_per_frame,
            score_norm=False,
            lm=CountingLanguageModel(lm_logp) if with_external_lm else None,
            lm_alpha=lm_alpha,
            lm_type="ilme",
            lm_beta=lm_beta,
        )[0]
        assert actual == expected, f"seed {seed}: got {actual}, DP optimum is {expected}"


@pytest.mark.parametrize("lm_beta", [0.0, 0.5])
@pytest.mark.parametrize("score_norm", [False, True])
def test_lodr_matches_exact_dp(lm_beta, score_norm):
    """The low-order LM has its own state, so this also checks it is re-ordered onto the parents."""
    nframes, vocab_size, max_tokens_per_frame, beam_width = 3, 3, 7, 256
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + 2201, (nframes, vocab_size, vocab_size))
        lm_logp = log_softmax(seed + 2301, (LM_POSITIONS, vocab_size, vocab_size))
        ilm_logp = internal_log_softmax(seed + 2401, (LM_POSITIONS, vocab_size, vocab_size))
        expected = solve_exact(
            logp,
            nframes,
            vocab_size,
            max_tokens_per_frame,
            score_norm,
            lm_logp=lm_logp,
            lm_alpha=0.6,
            ilm_logp=ilm_logp,
            lm_beta=lm_beta,
        )
        actual = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            beam_width=beam_width,
            max_tokens_per_frame=max_tokens_per_frame,
            score_norm=score_norm,
            lm=CountingLanguageModel(lm_logp),
            lm_alpha=0.6,
            lm_type="lodr",
            internal_lm=CountingLanguageModel(ilm_logp),
            lm_beta=lm_beta,
        )[0]
        assert actual == expected, f"seed {seed}: got {actual}, DP optimum is {expected}"


@pytest.mark.parametrize("lm_type", ["ilme", "lodr"])
def test_zero_beta_is_a_no_op(lm_type):
    """Subtracting the internal LM at beta=0 must decode identically to plain shallow fusion."""
    nframes, vocab_size = 4, 4
    model = build_model(vocab_size)
    for seed in range(4):
        logp = log_softmax(seed + 3131, (nframes, vocab_size, vocab_size))
        lm_logp = log_softmax(seed + 3232, (LM_POSITIONS, vocab_size, vocab_size))
        common = dict(beam_width=16, max_tokens_per_frame=2, lm_alpha=0.5)
        shallow = decode(model, logp[None], [nframes], vocab_size, nframes, lm=CountingLanguageModel(lm_logp), **common)[0]
        extra = dict(internal_lm=CountingLanguageModel(internal_log_softmax(seed + 3333, (LM_POSITIONS, vocab_size, vocab_size))))
        if lm_type == "ilme":
            extra = dict(ilm_logp=internal_log_softmax(seed + 3333, (vocab_size, vocab_size)))
        corrected = decode(
            model,
            logp[None],
            [nframes],
            vocab_size,
            nframes,
            lm=CountingLanguageModel(lm_logp),
            lm_type=lm_type,
            lm_beta=0.0,
            **extra,
            **common,
        )[0]
        assert shallow == corrected, f"seed {seed}: beta=0 changed the hypothesis"


@pytest.mark.parametrize("lm_type", ["ilme", "lodr"])
def test_internal_lm_subtraction_changes_the_hypothesis(lm_type):
    """Guards against the subtraction silently not reaching the scores at all."""
    nframes, vocab_size = 4, 4
    model = build_model(vocab_size)
    changed = 0
    for seed in range(8):
        logp = log_softmax(seed + 4141, (nframes, vocab_size, vocab_size))
        lm_logp = log_softmax(seed + 4242, (LM_POSITIONS, vocab_size, vocab_size))
        common = dict(beam_width=16, max_tokens_per_frame=2, lm_alpha=0.5, lm=CountingLanguageModel(lm_logp))
        baseline = decode(model, logp[None], [nframes], vocab_size, nframes, **common)[0]
        extra = dict(internal_lm=CountingLanguageModel(internal_log_softmax(seed + 4343, (LM_POSITIONS, vocab_size, vocab_size), scale=4.0)))
        if lm_type == "ilme":
            extra = dict(ilm_logp=internal_log_softmax(seed + 4343, (vocab_size, vocab_size), scale=4.0))
        corrected = decode(model, logp[None], [nframes], vocab_size, nframes, lm_type=lm_type, lm_beta=1.5, **extra, **common)[0]
        changed += baseline != corrected
    assert changed > 0, f"{lm_type} never altered the output"


def test_rejects_bad_lm_type_and_missing_internal_lm():
    model = build_model(3)
    stub_transducer(model, log_softmax(0, (1, 2, 3, 3)), [2], 3, 2)
    with pytest.raises(ValueError, match="lm_type"):
        model.recognize_beam(inputs=make_inputs(1), beam_width=4, lm_type="density_ratio")
    with pytest.raises(ValueError, match="internal_lm"):
        model.recognize_beam(inputs=make_inputs(1), beam_width=4, lm_type="lodr")


# --------------------------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------------------------


class _DecoderConfigStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TokenizerStub:
    def __init__(self, decoder_config):
        self.decoder_config = decoder_config


def _lm_config(**kwargs):
    """A `configs.LanguageModelConfig`, which now carries only the two model blobs."""
    return LanguageModelConfig(kwargs)


def test_beam_decoding_kwargs_from_decoder_config():
    model = build_model(4)

    # beam_width <= 0 means "no beam search", which is the default in every shipped config
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(beam_width=0, norm_score=True))
    assert model.get_beam_decoding_kwargs() == {}

    model.tokenizer = _TokenizerStub(_DecoderConfigStub(beam_width=7, norm_score=False))
    assert model.get_beam_decoding_kwargs() == {"beam_width": 7, "score_norm": False}

    # the LM only joins in once one is attached
    blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(beam_width=3, norm_score=True, lm_alpha=0.4))
    model.make_lm(_lm_config(external_config=blob))
    assert model.get_beam_decoding_kwargs() == {"beam_width": 3, "score_norm": True, "lm": model.lm, "lm_alpha": 0.4}


def test_beam_decoding_kwargs_carries_the_internal_lm_settings():
    """How to score comes from `decoder_config`; which models to score with, from `lm_config`."""
    model = build_model(4)
    blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    common = dict(beam_width=3, norm_score=True, lm_alpha=0.4, lm_beta=0.2)

    # no `lm_type` at all, ie. every config written before internal LM subtraction existed
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(**common))
    model.make_lm(_lm_config(external_config=blob))
    assert model.get_beam_decoding_kwargs() == {"beam_width": 3, "score_norm": True, "lm": model.lm, "lm_alpha": 0.4}

    model.tokenizer = _TokenizerStub(_DecoderConfigStub(lm_type="ilme", **common))
    model.make_lm(_lm_config(external_config=blob))
    assert model.get_beam_decoding_kwargs() == {
        "beam_width": 3,
        "score_norm": True,
        "lm": model.lm,
        "lm_alpha": 0.4,
        "lm_type": "ilme",
        "lm_beta": 0.2,
    }

    model.tokenizer = _TokenizerStub(_DecoderConfigStub(lm_type="lodr", **common))
    model.make_lm(_lm_config(external_config=blob, internal_config=blob))
    assert model.get_beam_decoding_kwargs()["internal_lm"] is model.internal_lm


def test_make_lm_builds_only_what_is_configured():
    """
    `make_lm` builds and nothing more -- it never reads `lm_type` and never judges the result.
    Whether the combination is coherent is `app_util.validate_lm`, tested separately below.
    """
    model = build_model(4)
    blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    internal_blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(1, (LM_POSITIONS, 4, 4))))
    model.tokenizer = _TokenizerStub(_DecoderConfigStub())

    model.make_lm(_lm_config(external_config=blob))
    assert isinstance(model.lm, CountingLanguageModel) and model.internal_lm is None

    model.make_lm(_lm_config(internal_config=internal_blob))
    assert model.lm is None and isinstance(model.internal_lm, CountingLanguageModel)

    model.make_lm(_lm_config(external_config=blob, internal_config=internal_blob))
    assert isinstance(model.lm, CountingLanguageModel) and isinstance(model.internal_lm, CountingLanguageModel)

    # an absent block is the same as an empty one, and neither is an error
    for empty in (_lm_config(), None):
        model.make_lm(empty)
        assert model.lm is None and model.internal_lm is None
    assert model.get_beam_decoding_kwargs() == {}

    # building must not depend on `lm_type` -- even a nonsensical pairing builds fine
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(lm_type="lodr"))
    model.make_lm(_lm_config(external_config=blob))
    assert isinstance(model.lm, CountingLanguageModel) and model.internal_lm is None

    # the config dicts must survive being built from
    assert "class_name" in blob


def test_validate_lm_raises_only_when_decoding_cannot_run():
    """LODR with no model to subtract is the one unrecoverable case."""
    model = build_model(4)
    blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    model.tokenizer = _TokenizerStub(_DecoderConfigStub())

    model.make_lm(_lm_config(external_config=blob))
    with pytest.raises(ValueError, match="lodr"):
        app_util.validate_lm(model, _DecoderConfigStub(beam_width=4, lm_type="lodr"))

    # every other combination is legal, however unwise, so it only warns
    app_util.validate_lm(model, _DecoderConfigStub(beam_width=4, lm_type="ilme", lm_alpha=0.3, lm_beta=0.1))
    app_util.validate_lm(model, _DecoderConfigStub(beam_width=4, lm_type="shallow", lm_alpha=0.3))
    app_util.validate_lm(model, _DecoderConfigStub(beam_width=0))

    model.make_lm(_lm_config(external_config=blob, internal_config=blob))
    app_util.validate_lm(model, _DecoderConfigStub(beam_width=4, lm_type="lodr", lm_alpha=0.3, lm_beta=0.1))

    # `DecoderConfig` still rejects an unknown type at config load, before any of this
    with pytest.raises(ValueError, match="lm_type"):
        DecoderConfig({"lm_type": "density_ratio"})


@pytest.mark.parametrize(
    "lm_type,external,internal,alpha,beta,lm_h5,expected",
    [
        ("shallow", True, False, 0.3, 0.0, "x.h5", None),  # the ordinary case, silent
        ("shallow", True, False, 0.3, 0.0, None, "untrained"),
        ("shallow", True, False, 0.0, 0.0, "x.h5", "lm_alpha is 0"),
        ("ilme", False, False, 0.0, 0.1, None, "nothing fused in"),
        ("ilme", True, True, 0.3, 0.1, "x.h5", "never reads it"),
        ("ilme", True, False, 0.3, 0.0, "x.h5", "lm_beta is 0"),
    ],
)
def test_validate_lm_warns_on_silent_misconfigurations(caplog, lm_type, external, internal, alpha, beta, lm_h5, expected):
    """Each of these decodes happily and just produces a worse WER, so a warning is the only signal."""
    model = build_model(4)
    blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    model.tokenizer = _TokenizerStub(_DecoderConfigStub())
    model.make_lm(_lm_config(external_config=blob if external else {}, internal_config=blob if internal else {}))

    with caplog.at_level(logging.WARNING):
        app_util.validate_lm(
            model,
            _DecoderConfigStub(beam_width=4, lm_type=lm_type, lm_alpha=alpha, lm_beta=beta),
            lm_h5=lm_h5,
            internal_lm_h5="y.h5",
        )
    warnings = " | ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    if expected is None:
        assert not warnings, f"expected no warning, got: {warnings}"
    else:
        assert expected in warnings, f"expected {expected!r} in: {warnings or '(none)'}"


def test_validate_lm_says_beam_search_is_off(caplog):
    """A configured LM with beam_width 0 is never used at all -- worth saying once, loudly."""
    model = build_model(4)
    blob = keras.saving.serialize_keras_object(CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    model.tokenizer = _TokenizerStub(_DecoderConfigStub())
    model.make_lm(_lm_config(external_config=blob))

    with caplog.at_level(logging.WARNING):
        app_util.validate_lm(model, _DecoderConfigStub(beam_width=0, lm_type="shallow", lm_alpha=0.3))
    assert any("beam search is off" in r.message for r in caplog.records)


def test_build_lm_loads_weights_passed_alongside_the_config(tmp_path):
    """Weights are an argument, not a config key -- the config says what the model is."""
    trained = CountingLanguageModel(log_softmax(7, (LM_POSITIONS, 4, 4)))
    trained.make()
    path = str(tmp_path / "lm.weights.h5")
    trained.save_weights(path)

    blob = keras.saving.serialize_keras_object(CountingLanguageModel(np.zeros((LM_POSITIONS, 4, 4), np.float32)))
    loaded = BaseModel.build_lm(blob, weights=path)
    np.testing.assert_allclose(loaded.table.numpy(), trained.table.numpy())

    # without a path the model keeps whatever its config gave it
    untouched = BaseModel.build_lm(blob)
    assert np.all(untouched.table.numpy() == 0.0)
    assert BaseModel.build_lm({}) is None and BaseModel.build_lm(None) is None


def test_make_lm_routes_each_weights_path_to_its_own_model(tmp_path):
    """`test.py` passes two h5 paths; each must land on the model it belongs to."""
    external = CountingLanguageModel(log_softmax(11, (LM_POSITIONS, 4, 4)))
    internal = CountingLanguageModel(log_softmax(22, (LM_POSITIONS, 4, 4)))
    external.make()
    internal.make()
    external_path, internal_path = str(tmp_path / "ext.weights.h5"), str(tmp_path / "int.weights.h5")
    external.save_weights(external_path)
    internal.save_weights(internal_path)

    empty = keras.saving.serialize_keras_object(CountingLanguageModel(np.zeros((LM_POSITIONS, 4, 4), np.float32)))
    model = build_model(4)
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(lm_type="lodr"))
    model.make_lm(
        _lm_config(external_config=empty, internal_config=empty),
        lm_weights=external_path,
        internal_lm_weights=internal_path,
    )

    np.testing.assert_allclose(model.lm.table.numpy(), external.table.numpy())
    np.testing.assert_allclose(model.internal_lm.table.numpy(), internal.table.numpy())
    assert not np.allclose(external.table.numpy(), internal.table.numpy()), "the two must be distinguishable"


@pytest.mark.parametrize("attribute", ["lm", "internal_lm"])
def test_language_model_is_not_tracked_as_a_keras_weight(attribute):
    """A fused LM must not leak its weights into the ASR model's checkpoint."""
    model = build_model(4)
    before = len(model.weights)
    setattr(model, attribute, CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4))))
    assert len(model.weights) == before
