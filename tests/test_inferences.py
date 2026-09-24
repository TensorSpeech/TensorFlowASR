"""
`tensorflow_asr.inferences.ASRInference`, over both of its backends.

Not to be confused with `tests/test_inference.py`, which drives `recognize` / `recognize_beam`
directly. This file is about the layer above them: one entry point that decodes either a live model
or an exported `.tflite`, in one pass or chunk by chunk, and the cache that makes the chunked form
continue where the previous call stopped.

The property worth the most here is that **chunked streaming reproduces a single whole-utterance
pass exactly**, through both backends. It holds only when every piece is right at once -- chunk
geometry read from the right place, the cache advancing by `signal_chunk_step` rather than
`signal_chunk_size`, and each new state fed back into the input it belongs to. On the `.tflite`
side that last one is the fragile part: the flatbuffer keeps only tensor *names*, and until
`make_tflite_function` began naming its input specs those names said nothing about which state was
which, so a Conformer silently fed its convolution cache into a subsampling slot. That is what
`test_state_feedback_pairs_every_state_output` and the equivalence tests pin down.

Models and builders are imported from the two neighbouring test modules rather than copied: the
streaming Conformer of `test_inference.py` is the only model here whose chunked decode is exactly
equal to its one-pass decode, and `test_tflite.py` already carries a builder per architecture.
"""

import numpy as np
import pytest

from tensorflow_asr import tf
from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.inferences import _INPUT_POSITION, ASRInference, StreamCache, _ordered, _signals
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

# A call takes a batch. Two rows rather than one so that a per-row slip -- a transcript built from
# the wrong row, or a state carried across rows -- has somewhere to show up.
BATCH = 2


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


def _inference(backend, model, exported):
    return ASRInference(model=model) if backend == "model" else ASRInference(tflite=exported)


@pytest.fixture
def model_asr(model):
    return ASRInference(model=model)


@pytest.fixture
def tflite_asr(exported):
    return ASRInference(tflite=exported)


@pytest.fixture
def asr(request, model, exported):
    """
    One `ASRInference` per test, over whichever backend the parametrisation asked for.

    Built fresh rather than shared: these objects hold a streaming session, and a test that leaves
    audio in the cache would otherwise decide what the next one sees. The expensive part -- the
    model and its conversion -- stays module scoped.
    """
    return _inference(request.param, model, exported)


@pytest.fixture
def single_asr(request, backends):
    """The same, built for one signal per call."""
    return _inference(request.param, *backends(1))


both_backends = pytest.mark.parametrize("asr", ["model", "tflite"], indirect=True)
both_backends_single = pytest.mark.parametrize("single_asr", ["model", "tflite"], indirect=True)


@pytest.fixture(scope="module")
def geometry(model):
    return tuple(int(value) for value in model.get_signal_chunk_size_and_step(1))


@pytest.fixture(scope="module")
def aligned_audio(geometry):
    """
    Audio covering a whole number of chunks exactly.

    The last chunk then ends flush with the signal, leaving only the `size - step` overlap behind,
    so `end()` has nothing new to flush and the streamed transcript can be compared with a
    one-pass decode without a padded tail confusing the two.
    """
    size, step = geometry
    return np.asarray(tf.random.stateless_normal([BATCH, size + step * 5], seed=[4, 5]) * 0.1, np.float32)


def width(audio):
    """Samples per row -- every row of a streaming batch always holds the same number."""
    return audio.shape[1]


def stream(asr, audio, piece=1000):
    """
    Feed `audio` in fixed pieces that have nothing to do with the chunk size, as a caller would.

    Returns one transcript per row, each the concatenation of every piece that row decoded.
    """
    asr.start()
    pieces = [asr(audio[:, index : index + piece], streaming=True) for index in range(0, width(audio), piece)]
    pieces.append(asr.end())
    return ["".join(row) for row in zip(*pieces)]


# --------------------------------------------------------------------------------- helpers


def test_signals_normalises_to_a_batch():
    """Callers hand over lists, tensors or arrays; a flat one is a batch of one, and `[B, T]` is kept."""
    values = [0.0, 0.5, -0.5, 1.0]

    single = _signals(np.asarray(values, np.float32))
    assert single.dtype == np.float32 and single.shape == (1, 4), "a flat vector is a batch of one"
    assert np.array_equal(_signals(values), single), "a plain list must convert"
    assert np.array_equal(_signals(tf.constant(values)), single), "a tensor must convert"
    assert np.array_equal(_signals(np.asarray(values, np.float64)), single), "float64 audio must be cast down"
    assert _signals(np.zeros([3, 4], np.float32)).shape == (3, 4), "a batch must be passed through untouched"


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


def test_geometry_comes_from_the_metadata_not_a_guess(model_asr, tflite_asr, model):
    """Both backends must agree with the model itself, or the two decode different chunks."""
    size, step = (int(value) for value in model.get_signal_chunk_size_and_step(1))

    assert model_asr._geometry() == (size, step)
    assert tflite_asr._geometry() == (size, step), "the exported metadata disagrees with the model"


# ------------------------------------------------------------------------------ one pass


@both_backends
def test_non_streaming_decodes_and_keeps_no_cache(asr, aligned_audio):
    """`streaming=False` is a whole-signal decode with fresh state that stores nothing."""
    transcripts = asr(aligned_audio, streaming=False)

    assert len(transcripts) == BATCH, "one transcript per row of the batch"
    assert all(isinstance(transcript, str) for transcript in transcripts)
    assert asr.cache == StreamCache(), "a non-streaming call must not leave a session behind"


def test_both_backends_agree_on_a_one_pass_decode(model_asr, tflite_asr, aligned_audio):
    """The exported graph is the same computation; a divergence here is an export defect."""
    assert model_asr(aligned_audio, streaming=False) == tflite_asr(aligned_audio, streaming=False)


@both_backends
def test_a_wrong_sized_batch_is_refused(asr, geometry):
    """
    The batch is part of the signature, so a mismatch is named rather than left to fail deeper.

    A model bakes it in at `make()` and an export at trace time. Unchecked, the model side reaches
    the decoder as a shape error from inside a `tf.while_loop` and the export side as an opaque
    interpreter resize failure.
    """
    size, _ = geometry

    with pytest.raises(ValueError, match="signal"):
        asr(np.zeros([BATCH + 1, size], np.float32), streaming=False)


def test_rows_are_decoded_independently(model_asr, aligned_audio):
    """
    A row's transcript must depend only on that row.

    The batch shares one set of carried states, one buffer and one invocation, so a slip in how
    those are folded -- a state gathered across the batch axis, a transcript read off the wrong
    row -- shows up as one row's audio changing another's transcript.

    The reference is that same row repeated to fill the batch, not the row decoded alone at batch
    1: `Transducer.recognize` sends a batch to `recognize_batch` and a single to
    `recognize_single`, and those two disagree, so comparing across them would fail for reasons
    that have nothing to do with this layer. Both sides here go through the batched decoder, which
    is what isolates cross-row leakage.
    """
    together = model_asr(aligned_audio, streaming=False)

    alone = []
    for row in range(BATCH):
        single = ASRInference(model=model_asr.model)
        single.batch_size = BATCH  # the model is built at BATCH; repeat one row to fill it
        repeated = np.repeat(aligned_audio[row : row + 1], BATCH, axis=0)
        alone.append(single(repeated, streaming=False)[0])

    assert together == alone, f"rows influenced each other:\n  batched ={together}\n  separate={alone}"


# ----------------------------------------------------------------------------- streaming


@both_backends_single
def test_streaming_reproduces_the_whole_utterance(single_asr, geometry):
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
    audio = np.asarray(tf.random.stateless_normal([1, size + step * 5], seed=[4, 5]) * 0.1, np.float32)

    whole = single_asr(audio, streaming=False)
    streamed = stream(single_asr, audio)

    assert streamed == whole, f"chunked decode diverged:\n  whole   ={whole!r}\n  streamed={streamed!r}"


@both_backends
def test_a_partial_chunk_is_buffered_and_decodes_nothing(asr, geometry):
    """Audio shorter than one chunk cannot be decoded, so it waits in the cache instead."""
    size, _ = geometry
    asr.start()

    assert asr(np.zeros([BATCH, size // 2], np.float32), streaming=True) == [""] * BATCH, "decoded before a chunk was full"
    assert width(asr.cache.signal) == size // 2, "the audio was dropped rather than buffered"
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
    asr(aligned_audio[:, :size], streaming=True)

    assert width(asr.cache.signal) == size - step
    assert np.array_equal(asr.cache.signal, aligned_audio[:, step:size]), "the retained tail is not the overlap"


@both_backends
def test_end_decodes_nothing_when_only_overlap_is_left(asr, geometry, aligned_audio):
    """
    Aligned audio leaves pure overlap, which was already decoded as the tail of the last chunk.

    Padding and decoding it again would append a phantom repeat to every transcript.
    """
    size, step = geometry
    asr.start()
    for index in range(0, width(aligned_audio), 1000):
        asr(aligned_audio[:, index : index + 1000], streaming=True)

    assert width(asr.cache.signal) == size - step, "this audio was meant to end flush with a chunk"
    assert asr.end() == [""] * BATCH, "overlap already decoded was decoded a second time"


@both_backends
def test_end_flushes_a_genuine_partial_tail(asr, geometry):
    """
    Audio left over beyond the overlap is real, undecoded speech and must not be thrown away.

    It is zero-padded up to a whole chunk because that is the only length the model accepts.
    """
    size, step = geometry
    audio = np.asarray(tf.random.stateless_normal([BATCH, size + step * 2 + 900], seed=[7, 8]) * 0.1, np.float32)
    asr.start()
    for index in range(0, width(audio), 1000):
        asr(audio[:, index : index + 1000], streaming=True)

    leftover = width(asr.cache.signal)
    assert leftover > size - step, "this audio was meant to leave a partial tail"
    assert asr.end() != [""] * BATCH, "the trailing audio was dropped instead of being padded and decoded"


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
    asr(aligned_audio[:, : size // 2], streaming=True)
    assert width(asr.cache.signal) > 0

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

    pieces = []
    start = 0
    while start + size <= width(aligned_audio):
        asr.start()
        pieces.append(asr(aligned_audio[:, start : start + size], streaming=True))
        asr.end()
        start += step
    independent = ["".join(row) for row in zip(*pieces)]

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

    asr = ASRInference(tflite=output)
    state_outputs = asr._outputs[2:]

    assert len(asr._feedback) == len(state_outputs), "some state output feeds nothing"
    fed = [position for position, _ in asr._feedback]
    assert fed == sorted(set(fed)), "a state input is fed twice"
    assert len(asr._states) - len(state_outputs) == (1 if name.startswith("ctc.") else 0), "unexpected number of unfed inputs"
    for position, _ in asr._feedback:
        assert 0 <= position < len(asr._states)


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
    size, _ = asr._geometry()

    audio = np.asarray(tf.random.stateless_normal([asr.batch_size, size * 3], seed=[1, 2]) * 0.1, np.float32)
    transcripts = stream(asr, audio, piece=size // 2)

    assert len(transcripts) == asr.batch_size
    assert all(isinstance(transcript, str) for transcript in transcripts)
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

    asr = ASRInference(tflite=output)
    seeded = [state for state in asr._tflite_initial_states() if state.dtype == np.float32 and list(state.shape) == [1, beam_width]]

    assert len(seeded) == 1, "the beam scores were not identified"
    assert seeded[0][0, 0] == 0, "the first hypothesis must start alive"
    assert np.all(seeded[0][0, 1:] == -1e9), "the remaining hypotheses must start dead"
