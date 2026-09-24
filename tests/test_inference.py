"""
Inference on a streaming Conformer transducer: ALSD++ beam search plus a KV-cached encoder.

The model here mirrors `examples/models/transducer/conformer/small-streaming.yml.j2` in kind but
not in size -- causal subsampling and convolution, chunked attention masking, and a bounded
attention memory -- shrunk to a few thousand parameters. The weights are random, so nothing here
asserts anything about transcription *quality*; what is checked is that the decoders honour their
contracts, that state actually flows between calls, and that the streaming properties the
architecture claims really hold.

Two of those properties are worth calling out, because both were measured rather than assumed:

* The encoder is *exactly* prefix-consistent when it is fully causal -- feeding the first half of
  a signal gives bit-for-bit the same frames as feeding all of it. Drop the chunk mask and that
  collapses immediately, since every frame then attends to the future. Both directions are
  pinned, so a config change that quietly de-streams the encoder fails here.

* Decoding the first chunk alone yields a strict prefix of the whole-utterance decode, under
  greedy and beam alike.

* Chunked *greedy* decoding is exactly equal to a single pass, convolutions included, provided
  each decode chunk covers a whole number of attention chunks. That holds only if every
  cross-chunk dependency is carried: the attention memory, the depthwise convolution's left
  context and the subsampling's per-layer left context. The alignment requirement is asserted from
  both sides -- aligned chunks match, misaligned ones do not.

* Chunked *beam* search carries the whole beam (`previous_beam_*` in `schemas.PredictInput`), so it
  no longer restarts from one hypothesis per chunk. It remains approximate regardless: tokens
  already emitted cannot be revised, while a single pass still can. That gap is asserted too, in a
  configuration where greedy streaming is exact so it cannot be confused with missing state.
"""

import inspect
import os

import numpy as np
import pytest

from tensorflow_asr import schemas, tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.tokenizers import CharTokenizer
from tensorflow_asr.utils import data_util

AUDIO_FILE_PATH = os.path.join(os.path.dirname(__file__), "test.flac")

# pad_end / preemphasis are pinned so that chunked feature extraction lines up with whole-signal
# extraction; otherwise every chunk gets zero-padded to a whole frame and the frames drift.
SPEECH_CONFIG = dict(
    sample_rate=16000,
    frame_ms=25,
    stride_ms=10,
    nfft=256,
    num_feature_bins=40,
    feature_type="log_mel_spectrogram",
    pad_end=False,
    preemphasis=0.0,
)

MEMORY_LENGTH = 8
CHUNK_SIZE = 4  # attention chunk, in encoder frames
NUM_HEADS, HEAD_SIZE, DMODEL = 2, 4, 8
NUM_BLOCKS = 2
SEED = 3


def subsampling_config(kernel=3):
    return {
        "class_name": "tensorflow_asr.models.layers.subsampling>Conv2dSubsampling",
        "config": {
            "filters": [8, 8],
            "kernels": [kernel, kernel],
            "strides": [2, 2],
            "paddings": ["causal", "causal"],
            "norms": ["batch", "batch"],
            "activations": ["swish", "swish"],
        },
    }


def build_model(tokenizer, streaming=True, memory_mode="kv", memory_length=MEMORY_LENGTH, seed=SEED, conv_kernel=3, dropout=0.1):
    """
    A tiny streaming Conformer transducer.

    `streaming=True` is what makes the encoder causal end to end: causal convolution padding,
    chunked attention with a bounded history, and a causal attention mask so no frame can see
    past its own position. Setting it False keeps the same weights but lets attention run over
    the whole utterance, which is the contrast used by the prefix-consistency tests.

    Note there is no separate history-size knob: `memory_length` is the left context for both
    mechanisms -- the mask when a whole utterance is processed at once, the cache when it arrives
    chunk by chunk. See `test_memory_length_is_the_only_left_context_knob`.
    """
    tf.keras.utils.set_random_seed(seed)
    streaming_kwargs = dict(encoder_chunk_size=CHUNK_SIZE, encoder_use_attention_causal_mask=True) if streaming else {}
    model = Conformer(
        blank=0,
        vocab_size=tokenizer.num_classes,
        speech_config=SPEECH_CONFIG,
        encoder_subsampling=subsampling_config(conv_kernel),
        encoder_dmodel=DMODEL,
        encoder_num_blocks=NUM_BLOCKS,
        encoder_head_size=HEAD_SIZE,
        encoder_num_heads=NUM_HEADS,
        encoder_kernel_size=conv_kernel,
        encoder_padding="causal",
        encoder_dropout=dropout,
        encoder_memory_length=memory_length,
        encoder_memory_mode=memory_mode,
        prediction_embed_dim=8,
        prediction_num_rnns=1,
        prediction_rnn_units=8,
        joint_dim=8,
        **streaming_kwargs,
    )
    model.tokenizer = tokenizer
    model.make(batch_size=1)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None}))
    tok.make()
    return tok


@pytest.fixture(scope="module")
def model(tokenizer):
    return build_model(tokenizer)


@pytest.fixture(scope="module")
def signal():
    """Two seconds of the fixture audio, shaped [1, T] as the predict signature expects."""
    raw = data_util.load_and_convert_to_wav(AUDIO_FILE_PATH)
    samples = data_util.read_raw_audio(raw)[: SPEECH_CONFIG["sample_rate"] * 2]
    return tf.reshape(samples, [1, -1])


def initial_input(model, signal):
    return schemas.PredictInput(
        inputs=signal,
        inputs_length=tf.shape(signal)[1:2],
        previous_tokens=model.get_initial_tokens(1),
        previous_encoder_states=model.get_initial_encoder_states(1),
        previous_decoder_states=model.get_initial_decoder_states(1),
    )


def decode(model, inputs, beam_width=0, **kwargs):
    """Route to ALSD++ beam search or greedy, the same way `make_tflite_function` does."""
    if beam_width > 0:
        return model.recognize_beam(inputs, beam_width=beam_width, **kwargs)
    return model.recognize(inputs, **kwargs)


def emitted(output):
    """The non-blank tokens, which is what a transcript is actually built from."""
    return [token for token in output.tokens.numpy()[0].tolist() if token != 0]


def encoder_frames(model, signal):
    features, features_length = model.feature_extraction((signal, tf.shape(signal)[1:2]), training=False)
    encoded, encoded_length, _ = model.encoder.call_next(features, features_length, model.get_initial_encoder_states(1))
    return encoded, int(encoded_length[0])


# --------------------------------------------------------------------------- output contract


@pytest.mark.parametrize("beam_width", [0, 2, 4])
def test_decoding_returns_a_valid_output(model, tokenizer, signal, beam_width):
    """Greedy and ALSD++ must both honour the PredictOutput contract."""
    output = decode(model, initial_input(model, signal), beam_width=beam_width)

    assert output.tokens.dtype == tf.int32
    assert output.tokens.shape[0] == 1
    assert bool(tf.reduce_all(output.tokens >= 0))
    assert bool(tf.reduce_all(output.tokens < tokenizer.num_classes)), "decoded a token outside the vocabulary"

    assert tuple(output.next_tokens.shape) == (1, 1), "next_tokens must be one token per utterance"
    assert output.next_encoder_states is not None, "a memory-enabled encoder must return states"
    assert output.next_decoder_states is not None

    transcript = tokenizer.detokenize(output.tokens).numpy()[0].decode()
    assert isinstance(transcript, str)


@pytest.mark.parametrize("beam_width", [0, 3])
def test_emissions_respect_the_per_frame_cap(model, signal, beam_width):
    """
    ALSD++ forces a blank once a hypothesis has emitted `max_tokens_per_frame` labels on a frame,
    so the transcript cannot exceed `frames * s`. Without that cap the lattice would allow
    unbounded emission on a single frame.
    """
    max_tokens_per_frame = 2
    _, frames = encoder_frames(model, signal)

    output = decode(model, initial_input(model, signal), beam_width=beam_width, max_tokens_per_frame=max_tokens_per_frame)

    assert len(emitted(output)) <= frames * max_tokens_per_frame, "emitted more labels than the per-frame cap allows"


def test_beam_search_is_deterministic(model, signal):
    """Two identical calls must agree; a beam that reorders under ties is not reproducible."""
    first = decode(model, initial_input(model, signal), beam_width=3)
    second = decode(model, initial_input(model, signal), beam_width=3)

    assert np.array_equal(first.tokens.numpy(), second.tokens.numpy())


def test_beam_search_matches_between_eager_and_graph(model, signal):
    """
    The beam search only ships if it traces: `make_tflite_function` wraps it in a `tf.function`.

    A `tf.while_loop` whose shape invariants are subtly wrong can still work eagerly and then
    produce different results once traced, so both are compared rather than just checked to run.
    """
    inputs = initial_input(model, signal)
    eager = decode(model, inputs, beam_width=3)

    @tf.function
    def traced(predict_input):
        return model.recognize_beam(predict_input, beam_width=3)

    graph = traced(inputs)

    assert np.array_equal(eager.tokens.numpy(), graph.tokens.numpy()), "graph mode disagreed with eager"


# --------------------------------------------------------------------------- kv cache


def test_kv_cache_states_are_projected_per_head(model, signal):
    """A kv-cached encoder carries [B, M, num_heads, head_size] per block, not [B, M, dmodel]."""
    output = decode(model, initial_input(model, signal), beam_width=2)
    blocks = output.next_encoder_states["blocks"]

    assert len(blocks) == NUM_BLOCKS
    for block_state in blocks:
        attention = block_state["attention"]
        assert set(attention) == {"key", "value"}, "expected a key and a value memory per block"
        for name, tensor in attention.items():
            assert tuple(tensor.shape) == (1, MEMORY_LENGTH, NUM_HEADS, HEAD_SIZE), f"{name}: {tuple(tensor.shape)}"


@pytest.mark.parametrize("beam_width", [0, 3])
def test_kv_and_hidden_memory_modes_decode_identically(tokenizer, signal, beam_width):
    """
    The point of the kv cache: at inference the weights are frozen, so caching the projections
    is a pure optimisation. Identical weights must therefore give identical transcripts, not
    merely similar ones -- see tests/test_memory.py for the layer-level version of this.
    """
    kv_model = build_model(tokenizer, memory_mode="kv")
    hidden_model = build_model(tokenizer, memory_mode="hidden")

    kv_tokens = decode(kv_model, initial_input(kv_model, signal), beam_width=beam_width).tokens.numpy()
    hidden_tokens = decode(hidden_model, initial_input(hidden_model, signal), beam_width=beam_width).tokens.numpy()

    assert np.array_equal(kv_tokens, hidden_tokens), "kv cache changed the decode versus the hidden-state cache"


def test_memory_advances_between_calls(model, signal):
    """The cache must roll forward, otherwise every chunk starts from an empty context."""
    first = decode(model, initial_input(model, signal), beam_width=0)
    before = tf.nest.flatten(model.get_initial_encoder_states(1))[0].numpy()
    after = tf.nest.flatten(first.next_encoder_states)[0].numpy()

    assert not np.allclose(before, after), "the encoder memory did not change after a call"


# --------------------------------------------------------------------------- streaming properties


def test_fully_causal_encoder_is_prefix_consistent(model, signal):
    """
    The defining property of a streaming encoder: frames already emitted never change.

    Feeding half the signal must reproduce the same frames as feeding all of it. This holds only
    because the configuration is causal throughout -- causal convolution padding plus a causal
    attention mask plus chunked history.
    """
    whole, _ = encoder_frames(model, signal)
    half, half_frames = encoder_frames(model, signal[:, : signal.shape[1] // 2])

    difference = np.abs(whole.numpy()[:, :half_frames] - half.numpy()[:, :half_frames]).max()
    assert difference < 1e-4, f"prefix frames changed when later audio was appended (max diff {difference:.3e})"


def test_non_streaming_encoder_is_not_prefix_consistent(tokenizer, signal):
    """
    The contrast that gives the test above its teeth.

    Without chunked/causal attention every frame attends over the whole utterance, so appending
    audio rewrites earlier frames. If this ever starts passing, the "streaming" configuration has
    stopped being the thing that makes streaming work and the test above proves nothing.
    """
    offline = build_model(tokenizer, streaming=False)
    whole, _ = encoder_frames(offline, signal)
    half, half_frames = encoder_frames(offline, signal[:, : signal.shape[1] // 2])

    difference = np.abs(whole.numpy()[:, :half_frames] - half.numpy()[:, :half_frames]).max()
    assert difference > 1e-3, "a non-causal encoder unexpectedly produced stable prefix frames"


def chunk_geometry(model, chunks=1):
    """
    Signal chunk covering a whole number of attention chunks.

    Delegates to the model rather than recomputing, so this exercises the real helper that callers
    use -- an arithmetic slip in `BaseModel.get_signal_chunk_size_and_step` would surface here as a
    broken streaming equivalence rather than hiding behind a duplicate implementation.
    """
    size, step = model.get_signal_chunk_size_and_step(chunks)
    return int(size), int(step)


@pytest.mark.parametrize("beam_width", [0, 3])
def test_first_chunk_decode_is_a_prefix_of_the_whole_utterance(model, signal, beam_width):
    """
    Streaming inference must not retract what it has already emitted.

    Decoding only the leading chunk has to yield a prefix of the full decode, for greedy and for
    ALSD++ alike. Note this is a property of the *first* chunk, where no cross-chunk convolution
    context is missing yet.
    """
    size, _ = chunk_geometry(model)
    whole = emitted(decode(model, initial_input(model, signal), beam_width=beam_width))
    first = emitted(decode(model, initial_input(model, signal[:, :size]), beam_width=beam_width))

    assert first, "the first chunk decoded nothing; the test would be vacuous"
    assert whole[: len(first)] == first, f"first chunk {first} is not a prefix of {whole[: len(first) + 3]}"


def stream(model, signal, beam_width):
    """Decode chunk by chunk, carrying tokens plus encoder and decoder state, as a server would."""
    size, step = chunk_geometry(model)
    tokens, previous = [], model.get_initial_tokens(1)
    encoder_states = model.get_initial_encoder_states(1)
    decoder_states = model.get_initial_decoder_states(1)

    start = 0
    while start + size <= int(signal.shape[1]):
        chunk = signal[:, start : start + size]
        output = decode(
            model,
            schemas.PredictInput(
                inputs=chunk,
                inputs_length=tf.shape(chunk)[1:2],
                previous_tokens=previous,
                previous_encoder_states=encoder_states,
                previous_decoder_states=decoder_states,
            ),
            beam_width=beam_width,
        )
        tokens += emitted(output)
        previous = output.next_tokens
        encoder_states = output.next_encoder_states
        decoder_states = output.next_decoder_states
        start += step

    return tokens


@pytest.mark.parametrize("beam_width", [0, 3])
def test_chunked_streaming_runs_and_stays_in_vocabulary(model, tokenizer, signal, beam_width):
    """A full chunk-by-chunk pass with state carry must complete and stay legal throughout."""
    tokens = stream(model, signal, beam_width)

    assert tokens, "streaming produced no tokens at all"
    assert all(0 <= token < tokenizer.num_classes for token in tokens)


def test_carrying_state_changes_the_result(model, signal):
    """
    Proves the carried state is actually consumed.

    Re-running each chunk from a fresh state must give a different answer; if it does not, the
    encoder and decoder states are being ignored and "streaming" is just independent chunks.
    """
    size, step = chunk_geometry(model)
    carried = stream(model, signal, beam_width=0)

    independent = []
    start = 0
    while start + size <= int(signal.shape[1]):
        chunk = signal[:, start : start + size]
        independent += emitted(decode(model, initial_input(model, chunk), beam_width=0))
        start += step

    assert carried != independent, "resetting the state per chunk changed nothing; state is not being used"


def synthetic_audio(model, chunks=5):
    size, step = chunk_geometry(model)
    return tf.random.stateless_normal([1, size + step * chunks], seed=[4, 5]) * 0.1


@pytest.mark.parametrize("conv_kernel", [1, 3])
def test_greedy_streaming_equals_whole_utterance(tokenizer, conv_kernel):
    """
    Chunked greedy decoding reproduces a single pass *exactly*, convolutions included.

    This is the strongest statement in the file: it holds only if every piece of streaming state
    is right at once -- the attention memory, the KV cache, the depthwise convolution's left
    context, the subsampling's per-layer left context, the decoder states and the carried last
    token. Any one of them wrong and the token sequences drift. `conv_kernel=1` is kept as a
    control: it removes the convolution dependency, so a failure there points at the attention or
    decoder state rather than the convolution caches.
    """
    model = build_model(tokenizer, conv_kernel=conv_kernel)
    audio = synthetic_audio(model)

    whole = emitted(decode(model, initial_input(model, audio), beam_width=0))
    streamed = stream(model, audio, beam_width=0)

    assert whole == streamed, f"chunked greedy diverged: whole={whole[:20]} streamed={streamed[:20]}"


def test_encoder_state_covers_attention_and_both_convolutions(model):
    """
    Every cross-chunk dependency has to appear in the encoder state, or streaming silently drifts.

    The subsampling keeps one cache per causal convolution; each block keeps its attention memory
    plus the depthwise convolution's left context.
    """
    state = model.get_initial_encoder_states(1)

    assert set(state) == {"subsampling", "blocks"}, f"unexpected encoder state keys: {sorted(state)}"
    assert len(state["subsampling"]) == 2, "one left-context cache per subsampling convolution"
    assert len(state["blocks"]) == NUM_BLOCKS
    for block_state in state["blocks"]:
        assert set(block_state) == {"attention", "convolution"}, f"block state missing an entry: {sorted(block_state)}"
        assert tuple(block_state["convolution"].shape)[1] == 2, "depthwise kernel 3 needs 2 frames of left context"


def test_misaligned_chunks_break_the_equivalence(tokenizer):
    """
    The equivalence is conditional on chunk alignment, and that is a real constraint on callers.

    A decode chunk covering half an attention chunk makes the whole-utterance pass group frames
    into `chunk_size` windows that the streamed pass never reproduces, so the two disagree however
    perfect the state carry is. Asserted so the requirement is discoverable rather than folklore.
    """
    model = build_model(tokenizer)
    frames = CHUNK_SIZE * model.time_reduction_factor // 2  # half an attention chunk
    size, step = (int(value) for value in model.feature_extraction.get_signal_chunk_size_and_step(frames))
    audio = synthetic_audio(model)

    whole = emitted(decode(model, initial_input(model, audio), beam_width=0))
    streamed, previous = [], model.get_initial_tokens(1)
    encoder_states = model.get_initial_encoder_states(1)
    decoder_states = model.get_initial_decoder_states(1)
    start = 0
    while start + size <= int(audio.shape[1]):
        chunk = audio[:, start : start + size]
        output = decode(
            model,
            schemas.PredictInput(
                inputs=chunk,
                inputs_length=tf.shape(chunk)[1:2],
                previous_tokens=previous,
                previous_encoder_states=encoder_states,
                previous_decoder_states=decoder_states,
            ),
            beam_width=0,
        )
        streamed += emitted(output)
        previous, encoder_states, decoder_states = output.next_tokens, output.next_encoder_states, output.next_decoder_states
        start += step

    assert whole != streamed, "misaligned chunks unexpectedly matched; the alignment rule may be unnecessary"


def test_streaming_beam_search_cannot_revise_committed_tokens(tokenizer):
    """
    Chunked beam search stays approximate even with the whole beam carried, and that is structural.

    Passing the beam state keeps all W hypotheses alive across a boundary -- see
    `test_beam_state_carries_every_hypothesis` -- but tokens already emitted for an earlier chunk
    cannot be taken back, whereas a whole-utterance run may still revise that prefix when a later
    frame makes another hypothesis win. Greedy composes because an argmax is a local decision;
    beam search does not.

    Asserted in a configuration where greedy streaming *is* exact, so the difference cannot be
    blamed on missing state -- this is the beam's lookahead, not the plumbing.
    """
    aligned = build_model(tokenizer)
    audio = synthetic_audio(aligned)

    whole = emitted(decode(aligned, initial_input(aligned, audio), beam_width=3))
    streamed = stream(aligned, audio, beam_width=3)
    greedy_whole = emitted(decode(aligned, initial_input(aligned, audio), beam_width=0))
    greedy_streamed = stream(aligned, audio, beam_width=0)

    assert greedy_whole == greedy_streamed, "greedy must still match; otherwise this proves nothing about the beam"
    assert whole != streamed, "chunked beam search matched the whole utterance; the beam may now be carried"


def test_convolution_context_is_what_breaks_multichunk_streaming(model, signal):
    """
    The known limitation, asserted so it cannot be mistaken for a passing equivalence.

    Only *attention* is cached across chunks. Neither the Conformer convolution module nor
    `Conv2dSubsampling` keeps any state, so at each boundary they pad with zeros instead of the
    previous chunk's frames and later chunks drift from a single pass.

    That this is the cause was established by elimination rather than assumed -- with kernel 3
    anywhere the decode diverges, and only when *both* the convolution module and the subsampling
    drop to kernel 1 does it become exact (the test above). Closing the gap for real needs a
    convolution cache alongside the attention memory.

    If this ever fails because the two now agree, that work has been done -- delete this test and
    promote the equivalence to the default configuration.
    """
    whole = emitted(decode(model, initial_input(model, signal), beam_width=0))
    streamed = stream(model, signal, beam_width=0)

    assert whole != streamed, "multi-chunk streaming now matches the whole utterance; see the docstring"


# --------------------------------------------------------------------------- streaming beam state

BEAM_WIDTH = 3


def beam_seed(model, beam_width=BEAM_WIDTH):
    return model.get_initial_beam_state(1, beam_width)


def decode_beam(model, audio, scores, last_tokens, states, beam_width=BEAM_WIDTH, **extra):
    return model.recognize_beam(
        schemas.PredictInput(
            inputs=audio,
            inputs_length=tf.shape(audio)[1:2],
            previous_tokens=model.get_initial_tokens(1),
            previous_encoder_states=model.get_initial_encoder_states(1),
            previous_decoder_states=model.get_initial_decoder_states(1),
            previous_beam_scores=scores,
            previous_beam_last_tokens=last_tokens,
            previous_beam_states=states,
            **extra,
        ),
        beam_width=beam_width,
    )


def test_beam_state_carries_every_hypothesis(model):
    """
    The whole beam crosses a chunk boundary, not just the winner.

    Without this the beam restarts from one hypothesis every chunk. The returned hypotheses must be
    genuinely distinct, otherwise carrying them is decoration.
    """
    audio = synthetic_audio(model)[:, : chunk_geometry(model)[0]]
    output = decode_beam(model, audio, *beam_seed(model))

    assert tuple(output.next_beam_scores.shape) == (1, BEAM_WIDTH)
    assert tuple(output.next_beam_last_tokens.shape) == (1, BEAM_WIDTH)
    assert tuple(output.next_beam_states.shape)[0] == BEAM_WIDTH, "decoder states fold the beam into the batch axis"

    scores = output.next_beam_scores.numpy()[0]
    assert len(set(scores.tolist())) > 1, f"all hypotheses scored identically: {scores}"


def test_beam_scores_are_rebased_on_the_best_hypothesis(model):
    """
    Scores are returned relative to the winner so they stay bounded over a long stream.

    Only differences between hypotheses affect any ranking, and the absolute log-probability would
    otherwise fall without limit and eventually underflow.
    """
    audio = synthetic_audio(model)[:, : chunk_geometry(model)[0]]
    output = decode_beam(model, audio, *beam_seed(model))
    scores = output.next_beam_scores.numpy()

    assert np.isclose(scores.max(), 0.0), f"the best hypothesis should sit at 0, got {scores.max()}"
    assert np.all(scores <= 0.0)


def test_resumed_beam_state_changes_the_decode(model):
    """
    Proves the carried beam is consumed rather than quietly replaced by a fresh seed.

    Seeding the live hypothesis with different last tokens must reach different transcripts, since
    the prediction network conditions on that token. Several tokens are tried because any single
    pair can coincide -- the argmax over a small vocabulary is not sensitive to every embedding.
    """
    audio = synthetic_audio(model)[:, : chunk_geometry(model)[0]]
    scores, _, states = beam_seed(model)

    decodes = set()
    for token in range(0, 29, 4):
        seeded = tf.concat([tf.fill([1, 1], token), tf.zeros([1, BEAM_WIDTH - 1], tf.int32)], axis=1)
        decodes.add(tuple(emitted(decode_beam(model, audio, scores, seeded, states))))

    assert len(decodes) > 1, "the resumed last tokens had no effect on any decode; the beam state is being ignored"


def test_moving_the_live_hypothesis_changes_the_decode(model):
    """The score vector selects which hypothesis is alive, so its slot must matter."""
    audio = synthetic_audio(model)[:, : chunk_geometry(model)[0]]
    _, _, states = beam_seed(model)
    dead = -1e9

    first = emitted(decode_beam(model, audio, tf.constant([[0.0, dead, dead]]), tf.constant([[7, 0, 0]], tf.int32), states))
    second = emitted(decode_beam(model, audio, tf.constant([[dead, 0.0, dead]]), tf.constant([[0, 7, 0]], tf.int32), states))

    assert first == second, "the same hypothesis in a different slot should decode identically"


def test_streaming_beam_carries_state_across_chunks(model):
    """A full chunked pass threading the beam must stay legal and keep the beam populated."""
    audio = synthetic_audio(model)
    size, step = chunk_geometry(model)
    scores, last_tokens, states = beam_seed(model)
    encoder_states = model.get_initial_encoder_states(1)
    decoder_states = model.get_initial_decoder_states(1)
    previous = model.get_initial_tokens(1)

    tokens, start, chunks = [], 0, 0
    while start + size <= int(audio.shape[1]):
        chunk = audio[:, start : start + size]
        output = model.recognize_beam(
            schemas.PredictInput(
                inputs=chunk,
                inputs_length=tf.shape(chunk)[1:2],
                previous_tokens=previous,
                previous_encoder_states=encoder_states,
                previous_decoder_states=decoder_states,
                previous_beam_scores=scores,
                previous_beam_last_tokens=last_tokens,
                previous_beam_states=states,
            ),
            beam_width=BEAM_WIDTH,
        )
        tokens += emitted(output)
        previous, encoder_states, decoder_states = output.next_tokens, output.next_encoder_states, output.next_decoder_states
        scores, last_tokens, states = output.next_beam_scores, output.next_beam_last_tokens, output.next_beam_states
        assert np.isclose(scores.numpy().max(), 0.0), "scores drifted off the rebased origin mid-stream"
        start += step
        chunks += 1

    assert chunks > 1 and tokens
    assert all(0 <= token < 29 for token in tokens)


def test_greedy_export_signature_is_unaffected_by_the_beam_fields(model):
    """
    Adding beam state must not change a greedy export -- the fields are emitted only for beams.

    A greedy tflite model already in production should keep the same inputs and outputs.
    """
    greedy = model.make_tflite_function(batch_size=1, beam_width=0).get_concrete_function()
    beam = model.make_tflite_function(batch_size=1, beam_width=BEAM_WIDTH).get_concrete_function()

    greedy_inputs = len(tf.nest.flatten(greedy.structured_input_signature))
    beam_inputs = len(tf.nest.flatten(beam.structured_input_signature))
    greedy_outputs = len(tf.nest.flatten(greedy.structured_outputs))
    beam_outputs = len(tf.nest.flatten(beam.structured_outputs))

    assert beam_inputs == greedy_inputs + 3, f"beam export should add exactly 3 inputs, got {beam_inputs - greedy_inputs}"
    assert beam_outputs == greedy_outputs + 3, f"beam export should add exactly 3 outputs, got {beam_outputs - greedy_outputs}"


# --------------------------------------------------------------------------- left context


def test_memory_length_is_the_only_left_context_knob(tokenizer):
    """
    One parameter for one receptive field.

    http://arxiv.org/abs/2010.11395 describes its history window as keeping "a fixed length of key
    and value vectors" -- masking and caching are two executions of the same left context, not two
    mechanisms to size independently. There used to be a separate `history_size` for the mask, and
    nothing tied it to the cache; a config could set them apart and decode the model under a
    receptive field it was never trained for.

    This asserts they cannot drift, by construction: `MultiHeadAttention` exposes no history knob,
    and its streaming mask reads `memory_length`.
    """
    from tensorflow_asr.models.layers.multihead_attention import MultiHeadAttention

    parameters = inspect.signature(MultiHeadAttention.__init__).parameters
    assert "history_size" not in parameters, "a second left-context knob reappeared"
    assert "memory_length" in parameters and "chunk_size" in parameters

    layer = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=HEAD_SIZE, memory_length=6, chunk_size=2, name="m")
    query = tf.random.normal([1, 8, DMODEL])
    mask = layer._compute_attention_mask(query, query)  # pylint: disable=protected-access

    # a query in the second chunk sees its own chunk plus `memory_length` frames before it
    attended = mask.numpy()[0][3]
    assert attended.sum() == 4, f"expected chunk(2) + as much history as exists, got {attended.sum()}"
    assert attended[:4].all() and not attended[4:].any(), f"unexpected span: {attended.astype(int)}"


@pytest.mark.parametrize("memory_length", [MEMORY_LENGTH // 2, MEMORY_LENGTH, MEMORY_LENGTH * 2])
def test_streaming_is_exact_only_at_the_trained_left_context(tokenizer, memory_length):
    """
    The cache must supply exactly what the mask assumed, which is now automatic.

    Before the merge these were separate numbers, and only their equality gave an exact streamed
    decode -- too short starved the cache, too long over-fed it. Both are now the same parameter,
    so varying it varies mask and cache together and the equivalence holds throughout. A regression
    that reintroduced a second knob would break this for the mismatched values.
    """
    model = build_model(tokenizer, memory_length=memory_length, conv_kernel=1)
    audio = synthetic_audio(model)

    whole = emitted(decode(model, initial_input(model, audio), beam_width=0))
    streamed = stream(model, audio, beam_width=0)

    assert whole == streamed, f"memory_length={memory_length}: whole={whole[:15]} streamed={streamed[:15]}"


def test_attention_memory_is_not_used_during_training(tokenizer):
    """
    Training must not consult the attention cache, in either mode.

    Training sees the whole utterance at once and gets its left context from the streaming mask, so
    the cache is redundant there -- and under `memory_mode="kv"` actively wrong, since cached
    projections are stale as soon as the weights move. The encoder therefore emits no `attention`
    entry under `training=True`, so a training step cannot silently consume one.

    Asserted on the state structure rather than by comparing outputs: `training=True` also switches
    dropout and BatchNorm, so two forward passes differ for reasons unrelated to the cache.

    The `convolution` entry is expected in both, deliberately. Unlike attention it has no masking
    equivalent -- a depthwise convolution simply needs its left context -- and it is inert during
    training anyway, since the training path never passes an initial state.
    """
    model = build_model(tokenizer, dropout=0.0)
    audio = synthetic_audio(model)[:, : chunk_geometry(model)[0]]
    features, features_length = model.feature_extraction((audio, tf.shape(audio)[1:2]), training=False)
    state = model.get_initial_encoder_states(1)

    inference = model.encoder((features, features_length), initial_state=state, training=False, return_states=True)[2]
    training = model.encoder((features, features_length), initial_state=state, training=True, return_states=True)[2]

    assert all("attention" in block for block in inference["blocks"]), "inference should carry the attention cache"
    assert all("attention" not in block for block in training["blocks"]), "the attention cache leaked into training"
    assert all("convolution" in block for block in training["blocks"]), "convolution context is needed in both"


def test_signal_chunk_size_walks_both_reductions_back(tokenizer):
    """
    `get_signal_chunk_size_and_step` must undo feature extraction *and* encoder subsampling.

    Missing either one gives a chunk that does not line up with an attention chunk, which is what
    breaks streaming equivalence (see `test_misaligned_chunks_break_the_equivalence`).
    """
    model = build_model(tokenizer)
    extraction = model.feature_extraction

    assert model.encoder.chunk_size == CHUNK_SIZE
    assert model.time_reduction_factor > 1, "the test is vacuous without subsampling"

    for nchunks in (1, 2, 3):
        size, step = model.get_signal_chunk_size_and_step(nchunks)
        frames = nchunks * CHUNK_SIZE * model.time_reduction_factor
        expected_size, expected_step = extraction.get_signal_chunk_size_and_step(frames)

        assert (int(size), int(step)) == (int(expected_size), int(expected_step))
        # step is short of size by exactly the window overlap
        assert int(size) - int(step) == extraction.frame_length - extraction.frame_step


def test_signal_chunk_yields_exactly_one_attention_chunk(tokenizer):
    """The returned samples must produce `chunk_size` encoder frames, not merely something close."""
    model = build_model(tokenizer)
    size, _ = model.get_signal_chunk_size_and_step(1)
    audio = tf.zeros([1, int(size)], tf.float32)

    _, frames = encoder_frames(model, audio)

    assert frames == CHUNK_SIZE, f"one chunk of audio gave {frames} encoder frames, expected {CHUNK_SIZE}"


def test_signal_chunk_falls_back_when_not_streaming(tokenizer):
    """
    A non-streaming encoder has no `chunk_size`, so one call is one encoder frame.

    Keeps the helper usable for frame-by-frame work instead of raising on a missing attribute.
    """
    model = build_model(tokenizer, streaming=False)
    assert model.encoder.chunk_size is None

    size, step = model.get_signal_chunk_size_and_step()
    expected = model.feature_extraction.get_signal_chunk_size_and_step(model.time_reduction_factor)

    assert (int(size), int(step)) == (int(expected[0]), int(expected[1]))
