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

import numpy as np
import pytest

from tensorflow_asr import keras, schemas, tf
from tensorflow_asr.models.decoders.language_model import LanguageModel
from tensorflow_asr.models.transducer.rnnt import RnnTransducer

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

    The state is that count. `score` is called *before* the next token is chosen, so an LM cannot
    condition on it -- conditioning comes from `previous_tokens`, and the state carries the
    recurrent part, exactly as the prediction network does. Making the distribution depend on the
    count means a beam that advanced the state on a blank, or failed to re-order states onto the
    selected parents, produces different scores and fails the comparison against the DP.
    """

    def __init__(self, table, **kwargs):
        super().__init__(**kwargs)
        self.table = tf.convert_to_tensor(table, dtype=tf.float32)  # [LM_POSITIONS, V, V]

    def get_initial_state(self, batch_size):
        return tf.zeros([batch_size, 1], dtype=tf.int32)

    def score(self, previous_tokens, previous_states):
        count = tf.minimum(tf.reshape(previous_states, [-1]), LM_POSITIONS - 1)
        indices = tf.stack([count, tf.reshape(previous_tokens, [-1])], axis=-1)
        return tf.gather_nd(self.table, indices), tf.reshape(previous_states, [-1, 1]) + 1


def log_softmax(seed, shape, scale=2.0):
    logits = np.random.RandomState(seed).randn(*shape).astype(np.float32) * scale
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def token_budget(nframes, max_tokens_per_frame, padded_frames=None):
    """`u` cap of the implementation: max_frames * 2 + 1, on the *padded* batch width."""
    return min(nframes * max_tokens_per_frame, 2 * (padded_frames or nframes) + 1)


def solve_exact(logp, nframes, vocab_size, max_tokens_per_frame, score_norm, lm_logp=None, lm_alpha=0.0, padded_frames=None):
    """
    Forward DP over `(t, u, last, expansions)` returning the best legal label sequence.

    Transitions are scored with eq. (3) of https://arxiv.org/abs/2506.00185 when `lm_logp` is
    given, and with the plain transducer log-probabilities otherwise.
    """
    max_u = token_budget(nframes, max_tokens_per_frame, padded_frames)

    def cost(t, u, last, token):
        # `u` doubles as the test LM's state: the number of labels emitted so far
        if token == BLANK:
            return (1.0 + lm_alpha) * logp[t, last, BLANK] if lm_logp is not None else logp[t, last, BLANK]
        if lm_logp is None:
            return logp[t, last, token]
        log_not_blank = np.log(-np.expm1(min(logp[t, last, BLANK], -1e-7)))
        return logp[t, last, token] + lm_alpha * (log_not_blank + lm_logp[min(u, LM_POSITIONS - 1), last, token])

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


def stub_transducer(model, logp, lengths, vocab_size, padded_frames):
    """Replace feature extraction, encoder and joint with a table lookup on (batch, frame, last)."""
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

    def call_next(current_frames, previous_tokens, previous_decoder_states):
        frame = tf.cast(tf.round(current_frames[:, 0, 0]), tf.int32)
        batch = tf.cast(tf.round(current_frames[:, 0, 1]), tf.int32)
        indices = tf.stack([batch, frame, tf.reshape(previous_tokens, [-1])], axis=-1)
        return tf.reshape(tf.gather_nd(table, indices), [-1, 1, 1, vocab_size]), previous_decoder_states

    model.call_next = call_next


def make_inputs(batch_size):
    return schemas.PredictInput(
        inputs=tf.zeros([batch_size, 16000]),
        inputs_length=tf.constant([16000] * batch_size, tf.int32),
        previous_tokens=tf.ones([batch_size, 1], tf.int32) * BLANK,
        previous_encoder_states=None,
        previous_decoder_states=tf.zeros([batch_size, 1, 1, 1]),
    )


def decode(model, logp, lengths, vocab_size, padded_frames, **kwargs):
    stub_transducer(model, logp, lengths, vocab_size, padded_frames)
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
# config plumbing
# --------------------------------------------------------------------------------------------


class _DecoderConfigStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TokenizerStub:
    def __init__(self, decoder_config):
        self.decoder_config = decoder_config


def test_beam_decoding_kwargs_from_decoder_config():
    model = build_model(4)

    # beam_width <= 0 means "no beam search", which is the default in every shipped config
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(beam_width=0, norm_score=True))
    assert model.get_beam_decoding_kwargs() == {}

    model.tokenizer = _TokenizerStub(_DecoderConfigStub(beam_width=7, norm_score=False))
    assert model.get_beam_decoding_kwargs() == {"beam_width": 7, "score_norm": False}

    # the LM only joins in once one is attached
    lm = CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4)))
    model.tokenizer = _TokenizerStub(_DecoderConfigStub(beam_width=3, norm_score=True, lm_alpha=0.4))
    model.lm = lm
    assert model.get_beam_decoding_kwargs() == {"beam_width": 3, "score_norm": True, "lm": lm, "lm_alpha": 0.4}


def test_language_model_is_not_tracked_as_a_keras_weight():
    """A fused LM must not leak its weights into the ASR model's checkpoint."""
    model = build_model(4)
    before = len(model.weights)
    model.lm = CountingLanguageModel(log_softmax(0, (LM_POSITIONS, 4, 4)))
    assert len(model.weights) == before
