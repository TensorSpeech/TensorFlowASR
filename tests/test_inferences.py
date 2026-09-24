"""
`tensorflow_asr.inferences`, over both of its backends: the shared `ASREngine` and the per-stream
`ASRInference` session on top of it.

Not to be confused with `tests/test_inference.py`, which drives `recognize` / `recognize_beam`
directly. This file is about the layer above them: one engine per loaded model that batches the
requests of many sessions, and one session per stream that decodes either in one pass or chunk by
chunk, with the cache that makes the chunked form continue where the previous call stopped.

The property worth the most here is that **chunked streaming reproduces a single whole-utterance
pass exactly**, through both backends. It holds only when every piece is right at once -- chunk
geometry read from the right place, the cache advancing by `signal_chunk_step` rather than
`signal_chunk_size`, and each new state fed back into the input it belongs to. On the `.tflite`
side that last one is the fragile part: the flatbuffer keeps only tensor *names*, and until
`make_tflite_function` began naming its input specs those names said nothing about which state was
which, so a Conformer silently fed its convolution cache into a subsampling slot. That is what
`test_state_feedback_pairs_every_state_output` and the equivalence tests pin down.

The second property is that **batching is invisible**: a session decoded in the same model call as
other sessions gets the same transcript as when it is decoded alone. It needs each session's state
cut out of and put back into the batch at the right rows, and padding that never leaks into a row.

Models and builders are imported from the two neighbouring test modules rather than copied: the
streaming Conformer of `test_inference.py` is the only model here whose chunked decode is exactly
equal to its one-pass decode, and `test_tflite.py` already carries a builder per architecture.
"""

import asyncio

import numpy as np
import pytest

from tensorflow_asr import tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.inferences import _INPUT_POSITION, ASREngine, ASRInference, StreamCache, _ordered, _signal
from tensorflow_asr.tokenizers import CharTokenizer
from tensorflow_asr.utils import app_util, tflite_util
from tests.test_inference import SPEECH_CONFIG, subsampling_config
from tests.test_inference import build_model as build_streaming_model
from tests.test_tflite import BUILDERS, convert
from tests.test_tflite import build_model as build_export_model

# `conv_kernel=1` removes the depthwise convolution's cross-chunk dependency. The equivalence still
# holds with a real kernel -- `tests/test_inference.py` asserts exactly that -- but keeping it out
# here means a failure points at this file's cache and state plumbing rather than at the encoder.
CONV_KERNEL = 1

# The batch the engine decodes per call. Two rather than one so that a per-slot slip -- a
# transcript built from the wrong row, or a state carried across rows -- has somewhere to show up.
BATCH = 2

# Long enough that requests submitted back to back always land in the same call.
BATCHING_WAIT_MS = 500


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None}))
    tok.make()
    return tok


@pytest.fixture(scope="module")
def backends(tokenizer, tmp_path_factory):
    """
    Builds a causal streaming Conformer and its export, at whatever batch size is asked for.

    A factory rather than a plain fixture because this file needs two batch sizes -- see
    `test_streaming_reproduces_the_whole_utterance` for why the equivalence has to be checked at
    one. Each pair is built once and reused, since conversion is the slow part.
    """
    built = {}

    def make(batch_size):
        if batch_size not in built:
            model = build_streaming_model(tokenizer, conv_kernel=CONV_KERNEL)
            model.make(batch_size=batch_size)
            output = str(tmp_path_factory.mktemp(f"tflite{batch_size}") / "streaming.tflite")
            app_util.convert_tflite(model=model, output=output, batch_size=batch_size, beam_width=0)
            built[batch_size] = (model, output)
        return built[batch_size]

    return make


@pytest.fixture(scope="module")
def model(backends):
    return backends(BATCH)[0]


@pytest.fixture(scope="module")
def exported(backends):
    return backends(BATCH)[1]


def _backend(backend, model, exported):
    """Keyword arguments that pick one backend, for `ASRInference` or `ASREngine`."""
    return {"model": model} if backend == "model" else {"tflite": exported}


@pytest.fixture
def backend(request, model, exported):
    return _backend(request.param, model, exported)


@pytest.fixture
def asr(backend):
    """
    One streaming session per test, over whichever backend the parametrisation asked for.

    Built fresh rather than shared: a session holds a stream, and a test that leaves audio in the
    cache would otherwise decide what the next one sees. The engine under it -- the loaded model --
    is shared, which is the point of it.
    """
    return ASRInference(**backend)


@pytest.fixture
def single_backend(request, backends):
    """The same backend choice, built for one signal per call."""
    return _backend(request.param, *backends(1))


both_backends = pytest.mark.parametrize("backend", ["model", "tflite"], indirect=True)
both_backends_single = pytest.mark.parametrize("single_backend", ["model", "tflite"], indirect=True)


@pytest.fixture(scope="module")
def geometry(model):
    return tuple(int(value) for value in model.get_signal_chunk_size_and_step(1))


def noise(length, seed):
    return np.asarray(tf.random.stateless_normal([length], seed=[seed, seed + 1]) * 0.1, np.float32)


@pytest.fixture(scope="module")
def aligned_audio(geometry):
    """
    Audio covering a whole number of chunks exactly.

    The last chunk then ends flush with the signal, leaving only the `size - step` overlap behind,
    so `end()` has nothing new to flush and the streamed transcript can be compared with a
    one-pass decode without a padded tail confusing the two.
    """
    size, step = geometry
    return noise(size + step * 5, seed=4)


def stream(asr, audio, piece=1000):
    """
    Feed `audio` in fixed pieces that have nothing to do with the chunk size, as a caller would.

    Returns the concatenation of every piece the session decoded.
    """
    asr.start()
    pieces = [asr(audio[index : index + piece]) for index in range(0, len(audio), piece)]
    return "".join(pieces) + asr.end()


async def astream(asr, audio, piece=1000):
    """`stream`, through the async methods, so several sessions can run at once in one loop."""
    asr.start()
    pieces = [await asr.infer(audio[index : index + piece]) for index in range(0, len(audio), piece)]
    return "".join(pieces) + await asr.aend()


class StepSpy:
    """Wraps an engine's `_step` to count the model calls and, optionally, to fail them."""

    def __init__(self, engine, error=None):
        self.calls = 0
        self.error = error
        self.step = engine._step
        engine._step = self

    def __call__(self, signals, lengths, states):
        self.calls += 1
        if self.error:
            raise self.error
        return self.step(signals, lengths, states)


# --------------------------------------------------------------------------------- helpers


def test_signal_normalises_to_one_stream():
    """Callers hand over lists, tensors or arrays, flat or as a single row; all become `[T]`."""
    values = [0.0, 0.5, -0.5, 1.0]

    single = _signal(np.asarray(values, np.float32))
    assert single.dtype == np.float32 and single.shape == (4,)
    assert np.array_equal(_signal(values), single), "a plain list must convert"
    assert np.array_equal(_signal(tf.constant(values)), single), "a tensor must convert"
    assert np.array_equal(_signal(np.asarray(values, np.float64)), single), "float64 audio must be cast down"
    assert np.array_equal(_signal(np.asarray([values], np.float32)), single), "a single row must be flattened"


@pytest.mark.parametrize("shape", [(2, 4), (1, 1, 4), ()])
def test_signal_refuses_anything_but_one_stream(shape):
    """A session is one stream, so several rows or a scalar are refused rather than reshaped."""
    with pytest.raises(ValueError, match="one stream"):
        _signal(np.zeros(shape, np.float32))


def test_ordered_refuses_an_export_whose_inputs_are_not_named():
    """
    An export made before `make_tflite_function` named its inputs must be rejected, not guessed at.

    Those files label their inputs `inputs`, `inputs_1`, ... in an order that is not the signature
    order, so pairing state by position would carry the wrong tensors and decode plausible-looking
    nonsense. The error names the fix rather than failing later inside the interpreter.
    """
    legacy = [{"name": "serving_default_inputs_1:0"}, {"name": "inputs_5"}]

    with pytest.raises(ValueError, match="Re-export"):
        _ordered(legacy, _INPUT_POSITION, "input")


def test_ordered_reads_the_position_out_of_a_named_input():
    """Names carry the signature position, and the interpreter's own ordering is ignored."""
    details = [
        {"name": tflite_util.INPUT_NAME.format(position=2)},
        {"name": "serving_default_" + tflite_util.INPUT_NAME.format(position=0) + ":0"},
        {"name": tflite_util.INPUT_NAME.format(position=1)},
    ]

    assert [detail["name"] for detail in _ordered(details, _INPUT_POSITION, "input")] == [
        "serving_default_tfasr_input_0:0",
        "tfasr_input_1",
        "tfasr_input_2",
    ]


# ---------------------------------------------------------------------------- construction


def test_requires_a_model_or_a_tflite():
    with pytest.raises(ValueError, match="Either"):
        ASRInference()
    with pytest.raises(ValueError, match="Either"):
        ASREngine()


def test_an_unbuilt_model_is_refused(tokenizer):
    """
    `make()` has to have run: the decoders read `self._batch_size` and the variables must exist.

    Caught when the object is built rather than on the first chunk of somebody's audio.
    """
    from tensorflow_asr.models.transducer.conformer import Conformer

    unbuilt = Conformer(
        blank=0,
        vocab_size=tokenizer.num_classes,
        speech_config=SPEECH_CONFIG,
        encoder_subsampling=subsampling_config(),
    )

    with pytest.raises(RuntimeError, match="make"):
        ASRInference(model=unbuilt)


def test_a_tflite_without_metadata_is_refused(tmp_path):
    """
    Chunk geometry is never guessed, so a file that cannot supply it is rejected outright.

    Converted here rather than through `app_util.convert_tflite`, which always writes the metadata.
    """
    module = tf.Module()
    module.identity = tf.function(lambda x: x, input_signature=[tf.TensorSpec([1], tf.float32)])
    foreign = tmp_path / "foreign.tflite"
    foreign.write_bytes(tf.lite.TFLiteConverter.from_concrete_functions([module.identity.get_concrete_function()], module).convert())

    with pytest.raises(ValueError, match="metadata"):
        ASRInference(tflite=str(foreign))


def test_geometry_comes_from_the_metadata_not_a_guess(model, exported):
    """Both backends must agree with the model itself, or the two decode different chunks."""
    size, step = (int(value) for value in model.get_signal_chunk_size_and_step(1))

    assert ASRInference(model=model).engine._geometry() == (size, step)
    assert ASRInference(tflite=exported).engine._geometry() == (size, step), "the exported metadata disagrees with the model"


# ------------------------------------------------------------------------- shared engine


@both_backends
def test_sessions_share_one_engine(backend):
    """Every session on one backend decodes on one loaded model, not one model per session."""
    first, second = ASRInference(**backend), ASRInference(**backend)

    assert first.engine is second.engine, "two sessions loaded two models"


@both_backends
def test_streaming_and_non_streaming_use_separate_engines(backend):
    """Whole signals are padded and chunks are not, so the two must never share a model call."""
    streaming = ASRInference(**backend, streaming=True).engine
    non_streaming = ASRInference(**backend, streaming=False).engine

    assert streaming is not non_streaming
    assert streaming.streaming and not non_streaming.streaming
    assert ASRInference(**backend, streaming=False).engine is non_streaming, "the non-streaming engine is not shared"


def test_different_backends_get_different_engines(model, exported, backends):
    assert ASRInference(model=model).engine is not ASRInference(tflite=exported).engine
    assert ASRInference(tflite=exported).engine is not ASRInference(tflite=backends(1)[1]).engine


@both_backends
def test_requests_from_several_sessions_share_one_model_call(backend, geometry):
    """Chunks that arrive together are decoded in one call, which is what batching is for."""
    size, _ = geometry
    engine = ASREngine(**backend, max_wait_ms=BATCHING_WAIT_MS)
    spy = StepSpy(engine)

    futures = [engine.submit(noise(size, seed=slot)) for slot in range(BATCH)]
    for future in futures:
        future.result()

    assert spy.calls == 1, f"{BATCH} requests took {spy.calls} model calls"


@both_backends
def test_more_requests_than_the_batch_are_all_decoded(backend, geometry):
    """A request that does not fit in the current call waits for the next one, it is not dropped."""
    size, _ = geometry
    engine = ASREngine(**backend, max_wait_ms=BATCHING_WAIT_MS)
    spy = StepSpy(engine)

    futures = [engine.submit(noise(size, seed=slot)) for slot in range(BATCH + 1)]
    transcripts = [future.result()[0] for future in futures]

    assert len(transcripts) == BATCH + 1 and all(isinstance(text, str) for text in transcripts)
    assert spy.calls == 2, "the batch was not filled before a second call was made"


@both_backends
def test_a_failed_call_raises_in_every_waiting_session(backend, geometry):
    """An error in the worker thread must reach the callers, not hang them or vanish."""
    size, _ = geometry
    engine = ASREngine(**backend, max_wait_ms=BATCHING_WAIT_MS)
    spy = StepSpy(engine, error=RuntimeError("model failed"))

    futures = [engine.submit(noise(size, seed=slot)) for slot in range(BATCH)]

    for future in futures:
        with pytest.raises(RuntimeError, match="model failed"):
            future.result(timeout=10)
    engine._step = spy.step  # a working step again: the worker must have survived the error
    assert isinstance(engine.submit(noise(size, seed=9)).result(timeout=60)[0], str)


# ------------------------------------------------------------------------------ one pass


@both_backends
def test_non_streaming_decodes_and_keeps_no_cache(backend, aligned_audio):
    """`streaming=False` is a whole-signal decode with fresh state that stores nothing."""
    asr = ASRInference(**backend, streaming=False)

    assert isinstance(asr(aligned_audio), str)
    assert asr.cache == StreamCache(), "a non-streaming call must not leave a session behind"


def test_both_backends_agree_on_a_one_pass_decode(model, exported, aligned_audio):
    """The exported graph is the same computation; a divergence here is an export defect."""
    assert ASRInference(model=model, streaming=False)(aligned_audio) == ASRInference(tflite=exported, streaming=False)(aligned_audio)


@both_backends
def test_a_padded_signal_decodes_as_it_does_alone(backend, geometry):
    """
    Whole signals of different lengths share a call by padding, and the padding must not show.

    Each row gets its real length in `inputs_length`, so the short one must decode exactly as it
    does in a call of its own. Both sides go through the same engine, so both are decoded at
    `BATCH` and the comparison measures only the padding.
    """
    size, step = geometry
    short, long = noise(size + step, seed=11), noise(size + step * 4, seed=12)
    engine = ASREngine(**backend, streaming=False, max_wait_ms=BATCHING_WAIT_MS)

    alone = [engine.submit(signal).result()[0] for signal in (short, long)]
    spy = StepSpy(engine)
    futures = [engine.submit(signal) for signal in (short, long)]
    together = [future.result()[0] for future in futures]

    assert spy.calls == 1, "the two signals were not decoded in one call"
    assert together == alone, f"padding changed a transcript:\n  together={together}\n  alone   ={alone}"


# ----------------------------------------------------------------------------- streaming


@both_backends_single
def test_streaming_reproduces_the_whole_utterance(single_backend, geometry):
    """
    The strongest statement in this file, and it holds for a causal model with aligned chunks.

    It needs every part right at once: the geometry, the cache advancing by `step` so consecutive
    chunks overlap by exactly the frame width, and every new state landing back on the input it
    came from. Get the last one wrong -- as positional pairing did on a Conformer -- and the
    transcripts drift apart without anything raising.

    Checked at one signal per call, because `Transducer.recognize` routes anything larger to
    `recognize_batch` instead of `recognize_single` and the two do not agree on identical audio --
    a batch of 2 decodes row 0 differently from that same row decoded alone, with no streaming
    involved. That is a property of the decoders, not of this layer, and until it is settled an
    equivalence asserted at batch 2 would be measuring the disagreement rather than the cache.
    """
    size, step = geometry
    audio = noise(size + step * 5, seed=4)

    whole = ASRInference(**single_backend, streaming=False)(audio)
    streamed = stream(ASRInference(**single_backend), audio)

    assert streamed == whole, f"chunked decode diverged:\n  whole   ={whole!r}\n  streamed={streamed!r}"


@both_backends
def test_sessions_streamed_together_decode_as_they_do_alone(backend, geometry):
    """
    Batching is invisible: a session gets the same transcript with or without company.

    More sessions than `BATCH` run at once, so some chunks share a call and some wait for the next
    one, and each session's state has to be cut out of the batch and put back at its own slot on
    every chunk. The reference is each session streamed alone on the same engine, which still
    decodes at `BATCH` with the other slots silent -- see
    `test_streaming_reproduces_the_whole_utterance` for why that matters.
    """
    size, step = geometry
    audios = [noise(size + step * (3 + index), seed=20 + index) for index in range(BATCH + 1)]

    alone = [stream(ASRInference(**backend), audio) for audio in audios]

    async def together():
        return await asyncio.gather(*(astream(ASRInference(**backend), audio) for audio in audios))

    assert asyncio.run(together()) == alone, "sessions decoded together influenced each other"


@both_backends
def test_async_and_sync_calls_agree(backend, aligned_audio):
    assert asyncio.run(astream(ASRInference(**backend), aligned_audio)) == stream(ASRInference(**backend), aligned_audio)
    non_streaming = ASRInference(**backend, streaming=False)
    assert asyncio.run(non_streaming.infer(aligned_audio)) == non_streaming(aligned_audio)


@both_backends
def test_a_partial_chunk_is_buffered_and_decodes_nothing(asr, geometry):
    """Audio shorter than one chunk cannot be decoded, so it waits in the cache instead."""
    size, _ = geometry
    asr.start()

    assert asr(np.zeros([size // 2], np.float32)) == "", "decoded before a chunk was full"
    assert len(asr.cache.signal) == size // 2, "the audio was dropped rather than buffered"
    assert asr.cache.states is None, "no chunk ran, so no state should have been created"


@both_backends
def test_only_the_overlap_is_kept_between_chunks(asr, geometry, aligned_audio):
    """
    After a chunk the cache holds `size - step` samples, and that is the next chunk's left context.

    Keeping `size` would decode the same audio twice; keeping nothing would drop the overlap that
    consecutive feature frames share.
    """
    size, step = geometry
    asr.start()
    asr(aligned_audio[:size])

    assert len(asr.cache.signal) == size - step
    assert np.array_equal(asr.cache.signal, aligned_audio[step:size]), "the retained tail is not the overlap"


@both_backends
def test_end_decodes_nothing_when_only_overlap_is_left(asr, geometry, aligned_audio):
    """
    Aligned audio leaves pure overlap, which was already decoded as the tail of the last chunk.

    Padding and decoding it again would append a phantom repeat to every transcript.
    """
    size, step = geometry
    asr.start()
    for index in range(0, len(aligned_audio), 1000):
        asr(aligned_audio[index : index + 1000])

    assert len(asr.cache.signal) == size - step, "this audio was meant to end flush with a chunk"
    assert asr.end() == "", "overlap already decoded was decoded a second time"


@both_backends
def test_end_flushes_a_genuine_partial_tail(asr, geometry):
    """
    Audio left over beyond the overlap is real, undecoded speech and must not be thrown away.

    It is zero-padded up to a whole chunk because that is the only length the model accepts.
    """
    size, step = geometry
    audio = noise(size + step * 2 + 900, seed=7)
    asr.start()
    for index in range(0, len(audio), 1000):
        asr(audio[index : index + 1000])

    assert len(asr.cache.signal) > size - step, "this audio was meant to leave a partial tail"
    assert asr.end() != "", "the trailing audio was dropped instead of being padded and decoded"


@both_backends
def test_end_clears_the_session(asr, aligned_audio):
    """`end()` closes the session, so the next `start()` begins from silence rather than mid-stream."""
    stream(asr, aligned_audio)

    assert asr.cache == StreamCache()


@both_backends
def test_start_abandons_a_half_finished_session(asr, geometry, aligned_audio):
    """A caller that gives up mid-stream must not have that audio prepended to the next session."""
    size, _ = geometry
    asr.start()
    asr(aligned_audio[: size // 2])
    assert len(asr.cache.signal) > 0

    asr.start()
    assert asr.cache == StreamCache(), "the abandoned audio survived into the new session"


@both_backends
def test_state_is_actually_carried_between_chunks(asr, aligned_audio, geometry):
    """
    Proves the cached state is consumed rather than merely stored.

    Decoding the same chunks from a fresh session each time must give a different answer; if it
    does not, "streaming" is just independent chunks and the equivalence above would be luck.
    """
    size, step = geometry
    carried = stream(asr, aligned_audio)

    independent = ""
    for start in range(0, len(aligned_audio) - size + 1, step):
        asr.start()
        independent += asr(aligned_audio[start : start + size])
        asr.end()

    assert carried != independent, "resetting state per chunk changed nothing; state is unused"


# ------------------------------------------------------------------- tflite state plumbing


@pytest.mark.parametrize("name", ["transducer.Conformer", "transducer.RnnTransducer", "ctc.Conformer"])
def test_state_feedback_pairs_every_state_output(name, tokenizer, tmp_path):
    """
    Every carried state must be paired with the output that replaces it, on every architecture.

    `transducer.Conformer` is the one that exposed the naming bug -- its encoder state is three
    leaves whose flatbuffer order differed from the outputs'. `ctc.Conformer` covers the other
    shape: CTC is not autoregressive, so `next_tokens` is None and never reaches the file, leaving
    one more state input than there are outputs. Pairing from the end is what absorbs that, and
    `previous_tokens` is the one input correctly left unfed.
    """
    output = str(tmp_path / "model.tflite")
    convert(build_export_model(name, tokenizer), batch_size=1, beam_width=0, output=output)

    engine = ASREngine(tflite=output)
    state_outputs = engine._outputs[2:]

    assert len(engine._feedback) == len(state_outputs), "some state output feeds nothing"
    fed = [position for position, _ in engine._feedback]
    assert fed == sorted(set(fed)), "a state input is fed twice"
    assert len(engine._states) - len(state_outputs) == (1 if name.startswith("ctc.") else 0), "unexpected number of unfed inputs"
    for position, _ in engine._feedback:
        assert 0 <= position < len(engine._states)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_architecture_streams(name, tokenizer, tmp_path):
    """
    Every exported architecture must load and stream, whatever its state structure.

    Correctness of the transcript is not the point here -- most of these are not causal, so their
    chunked decode legitimately differs from a single pass. What is asserted is that the plumbing
    holds: the states pair up, the interpreter accepts every tensor fed back, and the session ends
    clean.
    """
    output = str(tmp_path / "model.tflite")
    convert(build_export_model(name, tokenizer), batch_size=1, beam_width=0, output=output)
    asr = ASRInference(tflite=output)
    size, _ = asr.engine._geometry()

    transcript = stream(asr, noise(size * 3, seed=1), piece=size // 2)

    assert isinstance(transcript, str)
    assert asr.cache == StreamCache()


def test_beam_export_seeds_only_the_first_hypothesis(tokenizer, tmp_path):
    """
    A beam starts with one live hypothesis, not `beam_width` identical ones.

    Zero-filling the scores the way every other carried state is filled would leave W identical
    hypotheses, and the first expansion would then pick the same best token W times -- a beam that
    silently searches one path. The scores are found by shape because the flatbuffer carries no
    structure; `_tflite_initial_states` raises rather than guess if that is ever ambiguous.
    """
    beam_width = 2
    output = str(tmp_path / "beam.tflite")
    convert(build_export_model("transducer.RnnTransducer", tokenizer), batch_size=1, beam_width=beam_width, output=output)

    engine = ASREngine(tflite=output)
    seeded = [state for state in engine._tflite_initial_states() if state.dtype == np.float32 and list(state.shape) == [1, beam_width]]

    assert len(seeded) == 1, "the beam scores were not identified"
    assert seeded[0][0, 0] == 0, "the first hypothesis must start alive"
    assert np.all(seeded[0][0, 1:] == -1e9), "the remaining hypotheses must start dead"


def test_beam_export_streams_through_the_engine(tokenizer, tmp_path):
    """Beam states are `[B * W, ...]`, so a session's slot is W rows, and it must stream end to end."""
    output = str(tmp_path / "beam.tflite")
    convert(build_export_model("transducer.RnnTransducer", tokenizer), batch_size=1, beam_width=2, output=output)
    asr = ASRInference(tflite=output)
    size, _ = asr.engine._geometry()

    assert isinstance(stream(asr, noise(size * 3, seed=3), piece=size // 2), str)
    assert asr.cache == StreamCache()
