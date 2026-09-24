# Copyright 2026 Huy Le Nguyen (@nglehuy)
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

"""
Run a trained model, or an exported `.tflite`, over audio, for many sessions at once.

Two classes:

* `ASREngine` holds the loaded model. There is one engine per backend and mode in the process, so
  a server with many clients loads the model once. The engine has a worker thread that takes
  requests from all sessions and decodes up to `B` of them in one call. `B` is the batch size the
  model was built with at `make(batch_size=...)`, or the export was traced with.
* `ASRInference` is one session: one stream of audio, with its own state. Build one per websocket.
  Sessions built from the same `tflite` path or the same `model` object share one engine.

Two modes, chosen when the session is built:

* **Non-streaming** (`streaming=False`) sends the whole signal in one pass with fresh state and
  keeps nothing. This is exact for every architecture, offline ones included. Signals in one batch
  differ in length, so the engine pads them to the longest and passes each real length.
* **Streaming** (`streaming=True`) is fed one piece of audio at a time. Samples that do not yet
  fill a chunk stay in `self.cache` along with the session's decoder state, so the next call
  continues where this one stopped. `start()` opens a session and `end()` closes it -- `end()`
  zero-pads whatever is left to a whole chunk, decodes it, and returns that last piece of
  transcript. A session sends its chunks in order, one at a time, because each chunk starts from
  the state the previous one left. The engine batches chunks from different sessions, never two
  chunks of the same session.

Streaming and non-streaming use separate engines, so a padded whole signal never shares a call
with a streaming chunk.

Chunk geometry is never guessed. A `.tflite` carries it in its own metadata (see
`utils/tflite_util.py`), and a live model computes it with `BaseModel.get_signal_chunk_size_and_step`.
The two numbers differ because consecutive feature frames overlap: a call takes `signal_chunk_size`
samples but only advances by `signal_chunk_step`, so the tail of one chunk is the head of the next.

Chunked decoding only reproduces a whole-utterance pass when the model is causal and its chunks are
aligned -- `tests/test_inference.py` pins both directions. Streaming an offline model is legal here
and will simply decode worse.

Usage
-----

One pass over a whole file, from an exported model. The `.tflite` is self-describing, so nothing
else has to be loaded::

    from tensorflow_asr.inferences import ASRInference
    from tensorflow_asr.utils import data_util

    asr = ASRInference(tflite="/path/to/model.tflite", streaming=False)
    signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav("/path/to/audio.wav", sample_rate=16000))

    print(asr(signal))

The same from a checkpoint, built the way `tensorflow_asr test` builds it. `make()` must have run
and the tokenizer must be attached -- the model detokenizes its own output::

    import os

    from tensorflow_asr import tokenizers
    from tensorflow_asr.configs import Config
    from tensorflow_asr.utils import keras_util

    # `repodir` is not optional: the config is a jinja template and that is its include root.
    config = Config("/path/to/config.yml.j2", training=False, repodir=os.getcwd())
    tokenizer = tokenizers.get(config)
    tokenizer.make()

    model = keras_util.model_from_config(config.model_config)
    model.tokenizer = tokenizer
    model.make(batch_size=8)  # up to 8 sessions are decoded in one call
    model.load_weights("/path/to/weights.h5", skip_mismatch=False)

    asr = ASRInference(model=model, streaming=False)
    print(asr(signal))

Streaming, one piece of audio at a time. The pieces have nothing to do with the chunk size -- feed
whatever the source produces and the cache does the aligning. Each call returns only what it
decoded, so the caller appends::

    asr = ASRInference(tflite="/path/to/model.tflite")  # streaming=True is the default
    transcript = ""
    for block in microphone():  # any producer: a device, a socket, a file read in pieces
        transcript += asr(block)
    transcript += asr.end()  # pads the last partial chunk, decodes it, closes the session

In an async server, use `infer()` and `aend()` so a decode does not block the event loop. Every
websocket gets its own session, and all sessions share the engine::

    @app.websocket("/asr")
    async def asr_socket(websocket: WebSocket):
        await websocket.accept()
        asr = ASRInference(tflite="/path/to/model.tflite")
        try:
            while True:
                block = np.frombuffer(await websocket.receive_bytes(), dtype=np.float32)
                await websocket.send_text(await asr.infer(block))
        except WebSocketDisconnect:
            pass
        finally:
            await asr.aend()
"""

import asyncio
import concurrent.futures
import logging
import queue
import re
import threading
import time
import typing

import numpy as np

from tensorflow_asr import schemas, tf
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import file_util, tflite_util

logger = logging.getLogger(__name__)

# Position of a tensor in the exported signature, read off its name. `get_input_details()` and
# `get_output_details()` both come back in interpreter order, not signature order, and
# `get_signature_list()` is empty for some architectures (both Conformers), so the name is the only
# thing left to sort on.
#
# Inputs are named by `BaseModel.make_tflite_function` -- see `tflite_util.INPUT_NAME` for why they
# have to be. Outputs are numbered by the converter: `Identity` / `Identity_N` before the variables
# are frozen, and `StatefulPartitionedCall:N` or `PartitionedCall:N` after, which of the two
# depending on the architecture (`ctc.Jasper` produces the second).
_INPUT_POSITION = re.compile(tflite_util.INPUT_NAME.format(position=r"(\d+)") + r"(?::\d+)?$")
_OUTPUT_POSITION = re.compile(r"(?:Identity|(?:Stateful)?PartitionedCall)(?:[:_](\d+))?$")


def _ordered(details, pattern, kind):
    """Details sorted into signature order, by the position encoded in each name."""
    positions = {}
    for detail in details:
        match = pattern.search(detail["name"])
        if match is None:
            raise ValueError(
                f"cannot read the signature position of {kind} tensor {detail['name']!r}. Re-export this model with `tensorflow_asr tflite`."
            )
        positions[int(match.group(1) or 0)] = detail
    if sorted(positions) != list(range(len(details))):
        raise ValueError(f"{kind} tensors are not a contiguous signature: positions {sorted(positions)}")
    return [positions[index] for index in range(len(details))]


def _signal(signal) -> np.ndarray:
    """
    One stream's audio as a flat float32 vector, whatever container it arrived in.

    A session is one stream, so `[T]` and `[1, T]` are the only shapes accepted. Anything else is
    rejected rather than reshaped, since there is no reading of several rows that is obviously
    right for a single session.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 2 and signal.shape[0] == 1:
        signal = signal[0]
    if signal.ndim != 1:
        raise ValueError(f"a session is one stream: expected [samples] or [1, samples], got shape {signal.shape}")
    return signal


def _take(leaf, row, batch_size):
    """
    The rows of one batch slot out of a state leaf.

    A leaf is `[batch_size * k, ...]`: k is 1 for most states and `beam_width` for the beam ones,
    which are `[B * W, ...]` with each row's hypotheses kept together. The Keras mask a memory
    state carries is sliced along with it, or the next chunk would treat the memory as padding.
    """
    if leaf is None:
        return None
    k = int(leaf.shape[0]) // batch_size
    part = leaf[row * k : (row + 1) * k]
    mask = getattr(leaf, "_keras_mask", None)
    if mask is not None:
        part._keras_mask = mask[row * k : (row + 1) * k]  # pylint: disable=protected-access
    return part


def _concat(*leaves):
    """The inverse of `_take`: the slots of one state leaf stacked back into a batch."""
    if leaves[0] is None:
        return None
    if isinstance(leaves[0], np.ndarray):
        return np.concatenate(leaves)
    joined = tf.concat(leaves, 0)
    masks = [getattr(leaf, "_keras_mask", None) for leaf in leaves]
    if all(mask is not None for mask in masks):
        joined._keras_mask = tf.concat(masks, 0)  # pylint: disable=protected-access
    return joined


def _split(states, batch_size):
    """A batch state as one state per slot."""
    return [tf.nest.map_structure(lambda leaf, row=row: _take(leaf, row, batch_size), states) for row in range(batch_size)]


def _join(states):
    """One state per slot as a batch state."""
    return tf.nest.map_structure(_concat, *states)


class StreamCache(typing.NamedTuple):
    """
    What a streaming session carries from one call to the next, on `ASRInference.cache`.

    `signal` is a `[n]` buffer: audio that has arrived but not yet filled a chunk, plus the tail of
    the last chunk that the next one overlaps. `states` is this session's slot of whatever the
    backend threads through -- a list of arrays for a `.tflite`, the trailing fields of
    `schemas.PredictInput` for a live model -- and is deliberately opaque, so one loop drives both.

    Both are `None` until the session has something to remember: no audio buffered, and no state
    because nothing has decoded yet.

    Being a tuple it is replaced rather than edited, so `start()` and `end()` reset by assigning an
    empty one.
    """

    signal: typing.Optional[np.ndarray] = None
    states: typing.Any = None


class ASREngine:
    """
    One loaded model shared by every session, with a worker thread that batches their requests.

    Get one with `ASREngine.get`, which returns the same engine for the same backend and mode. A
    request is one session's signal and its state. The worker takes the first request in the queue,
    waits up to `max_wait_ms` for more, and decodes up to `batch_size` of them in one call. Slots
    with no request decode silence from the initial state, and their result is thrown away.

    Streaming and non-streaming requests go to separate engines. Streaming chunks are all
    `signal_chunk_size` long, so they stack without padding. Whole signals differ in length, so they
    are zero-padded to the longest in the batch and each row gets its real length in
    `inputs_length`. Keeping the two apart means a padded tail never reaches a carried state.

    The worker thread is the only code that runs the backend after setup, so no lock is needed
    around it. The two engines of one live model do call that model from two threads.
    """

    _engines = {}
    _engines_lock = threading.Lock()

    @classmethod
    def get(cls, tflite: str = None, model: BaseModel = None, streaming: bool = True, max_wait_ms: float = 5):
        """
        The shared engine for this backend and mode, built on the first call.

        A `.tflite` is keyed by its path and a model by its identity. `max_wait_ms` only applies to
        the call that builds the engine.
        """
        if not tflite and not model:
            raise ValueError("Either `tflite` or `model` must be provided.")
        key = (file_util.preprocess_paths(tflite) if tflite else id(model), streaming)
        with cls._engines_lock:
            if key not in cls._engines:
                cls._engines[key] = cls(tflite=tflite, model=model, streaming=streaming, max_wait_ms=max_wait_ms)
            return cls._engines[key]

    def __init__(self, tflite: str = None, model: BaseModel = None, streaming: bool = True, max_wait_ms: float = 5):
        if not tflite and not model:
            raise ValueError("Either `tflite` or `model` must be provided.")
        self.tflite = tflite
        self.model = model
        self.streaming = streaming
        self.max_wait = max_wait_ms / 1000
        # Loading the interpreter and reading the metadata is done once, here, rather than on the
        # first chunk: a missing file or an export too old to carry metadata should fail when the
        # object is built, not part way through someone's audio stream.
        if tflite:
            self._setup_tflite()
            self._step, initial_states = self._tflite_step, self._tflite_initial_states()
        else:
            self._setup_model()
            self._step, initial_states = self._model_step, self._model_initial_states()
        # One slot's starting state. Every slot starts the same, so the first is as good as any.
        self.initial_state = _split(initial_states, self.batch_size)[0]
        self._requests = queue.Queue()
        threading.Thread(target=self._serve, name="ASREngine", daemon=True).start()

    def submit(self, signal: np.ndarray, state=None) -> concurrent.futures.Future:
        """
        Queue one `[n]` signal for decoding, from `state` or from the initial state if None.

        The future resolves to `(transcript, next_state)`.
        """
        future = concurrent.futures.Future()
        self._requests.put((signal, state, future))
        return future

    def _serve(self):
        while True:
            batch = [self._requests.get()]
            deadline = time.monotonic() + self.max_wait
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._requests.get(timeout=max(0.0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            self._run(batch)

    def _run(self, batch):
        """Decode up to `batch_size` requests in one call and resolve each one's future."""
        try:
            width = max(len(signal) for signal, _, _ in batch)
            signals = np.zeros([self.batch_size, width], dtype=np.float32)
            lengths = np.full([self.batch_size], width, dtype=np.int32)
            states = [self.initial_state] * self.batch_size
            for slot, (signal, state, _) in enumerate(batch):
                signals[slot, : len(signal)] = signal
                lengths[slot] = len(signal)
                if state is not None:
                    states[slot] = state
            transcripts, next_states = self._step(signals, lengths, _join(states))
            next_states = _split(next_states, self.batch_size)
        except Exception as error:  # pylint: disable=broad-except
            # Raised in the caller that waits on the future, not lost in this thread.
            for _, _, future in batch:
                future.set_exception(error)
            return
        for slot, (_, _, future) in enumerate(batch):
            future.set_result((transcripts[slot], next_states[slot]))

    def _geometry(self):
        """`(signal_chunk_size, signal_chunk_step)` -- samples per call, and samples per advance."""
        if self.tflite:
            return int(self.metadata["signal_chunk_size"]), int(self.metadata["signal_chunk_step"])
        size, step = self.model.get_signal_chunk_size_and_step(1)
        return int(size), int(step)

    # ------------------------------------ TFLITE ------------------------------------ #

    def _setup_tflite(self):
        import tensorflow_text as tft
        from tensorflow.lite.python import interpreter as tflite_interpreter

        self.metadata = tflite_util.read_metadata(self.tflite)
        if not self.metadata:
            raise ValueError(f"{self.tflite} carries no TFASR metadata. Re-export it with `tensorflow_asr tflite`.")

        # The text ops that produce the in-graph transcript are not TFLite builtins, so the plain
        # interpreter cannot load these models.
        self.interpreter = tflite_interpreter.InterpreterWithCustomOps(
            model_path=file_util.preprocess_paths(self.tflite),
            custom_op_registerers=tft.tflite_registrar.SELECT_TFTEXT_OPS,
        )
        self._allocated = None
        self._locate_tensors()
        # Baked into the graph when the export was traced, so it is the batch every call must bring.
        self.batch_size = int(self._signal["shape_signature"][0])

    def _locate_tensors(self):
        """
        Sort the tensors into signature order and pair each new state with the old one it replaces.

        Signature order is `inputs, inputs_length, previous_tokens, encoder states..., decoder
        states...` going in and `transcript, tokens, next_tokens, encoder states..., decoder
        states...` coming out, so the audio is the first two inputs, the transcript the first
        output, and the carried state is the tail of both lists. A beam export appends
        `previous_beam_*` to one tail and `next_beam_*` to the other, and stays lined up.

        The two tails are paired from the *end* rather than the start, because a model can drop an
        output: CTC is not autoregressive, so its `next_tokens` is None, never reaches the
        flatbuffer, and leaves one more state input than there are state outputs. Pairing from the
        end carries the states correctly and leaves `previous_tokens` at the blank it started on,
        which is what a decoder with no token feedback wants anyway. Every pair is then checked for
        a compatible dtype and shape, so an architecture that breaks this fails here rather than by
        quietly decoding with the wrong tensors.

        That check reads `shape_signature`, not `shape`. An output whose length the tracer could
        not prove carries a -1 there and a placeholder 1 in `shape`, so a Conformer's returned
        left-context cache advertises [1, 1, 8] against the [1, 2, 8] it feeds. It really is 2
        frames once invoked -- the -1 is missing static knowledge, not a different tensor.
        """
        self._inputs = _ordered(self.interpreter.get_input_details(), _INPUT_POSITION, "input")
        self._outputs = _ordered(self.interpreter.get_output_details(), _OUTPUT_POSITION, "output")
        self._signal, self._length = self._inputs[0], self._inputs[1]
        self._states = self._inputs[2:]

        state_outputs = self._outputs[2:]  # everything after the transcript and the emitted tokens
        if len(state_outputs) > len(self._states):
            raise ValueError(f"{len(state_outputs)} state outputs cannot feed {len(self._states)} state inputs")
        offset = len(self._states) - len(state_outputs)
        for position, detail in enumerate(state_outputs):
            carried = self._states[offset + position]
            produced, expected = list(detail["shape_signature"]), list(carried["shape_signature"])
            compatible = len(produced) == len(expected) and all(a in (-1, b) or b == -1 for a, b in zip(produced, expected))
            if detail["dtype"] != carried["dtype"] or not compatible:
                raise ValueError(
                    f"state output {detail['name']} {produced} cannot feed input {carried['name']} {expected}; the signature tails do not line up"
                )
        # Which carried state each output replaces, as (position in the state list, tensor index).
        self._feedback = [(offset + position, detail["index"]) for position, detail in enumerate(state_outputs)]

    def _tflite_initial_states(self):
        """Zeros for a carried state, the blank id for a carried token -- one value per state input."""
        states = []
        for detail in self._states:
            fill = self.metadata["blank"] if np.issubdtype(detail["dtype"], np.integer) else 0
            states.append(np.full(detail["shape"], fill, dtype=detail["dtype"]))

        # A beam does not start with W live hypotheses: they are identical, so the first expansion
        # would pick the same best token W times. Only the first is alive and the rest are held at
        # -1e9, matching `BaseModel.get_initial_beam_state`. The scores are the one state that
        # cannot be zero-filled, and with no structure in the flatbuffer they are found by shape:
        # [batch, beam_width] and floating point, which no other state input has.
        beam_width = int(self.metadata.get("beam_width", 0) or 0)
        if beam_width > 1:
            batch = int(self._signal["shape_signature"][0])
            scores = [
                index
                for index, detail in enumerate(self._states)
                if np.issubdtype(detail["dtype"], np.floating) and list(detail["shape"]) == [batch, beam_width]
            ]
            if len(scores) != 1:
                raise ValueError(f"expected exactly one [{batch}, {beam_width}] float input to seed the beam scores, found {len(scores)}")
            states[scores[0]][:, 1:] = -1e9
        return states

    def _tflite_step(self, signals, lengths, states):
        """One invocation on the batch. Returns a transcript per row, and the next state."""
        if self._allocated != signals.shape[1]:
            self.interpreter.resize_tensor_input(self._signal["index"], list(signals.shape), strict=True)
            self.interpreter.allocate_tensors()
            self._allocated = signals.shape[1]
            self._locate_tensors()  # allocation rebuilds the descriptors, tensor indices included

        self.interpreter.set_tensor(self._signal["index"], signals)
        self.interpreter.set_tensor(self._length["index"], lengths.astype(self._length["dtype"]))
        for detail, value in zip(self._states, states):
            self.interpreter.set_tensor(detail["index"], value)

        self.interpreter.invoke()

        transcripts = [row.decode("utf-8") for row in self.interpreter.get_tensor(self._outputs[0]["index"])]
        # `get_tensor` copies, so the carried state survives the next call overwriting the buffers.
        next_states = list(states)
        for position, tensor_index in self._feedback:
            next_states[position] = self.interpreter.get_tensor(tensor_index)
        return transcripts, next_states

    # ------------------------------------- MODEL ------------------------------------- #

    def _setup_model(self):
        if not self.model.built:
            raise RuntimeError("Call `model.make(batch_size=...)` before running inference on it.")
        # Set by `make()`, and the decoders branch on it themselves -- `Transducer.recognize` picks
        # its single-utterance path at 1 -- so it is the batch every call must bring.
        self.batch_size = int(self.model._batch_size)
        # Empty unless `decoder_config.beam_width` is positive, which is how every shipped config
        # ships. Read once: it also carries the language model, and rebuilding it per chunk would
        # re-read the config on every call.
        self._beam_kwargs = self.model.get_beam_decoding_kwargs()

    def _model_initial_states(self):
        """The trailing fields of `PredictInput`, as keyword arguments to spread into it."""
        states = {
            "previous_tokens": self.model.get_initial_tokens(self.batch_size),
            "previous_encoder_states": self.model.get_initial_encoder_states(self.batch_size),
            "previous_decoder_states": self.model.get_initial_decoder_states(self.batch_size),
        }
        beam_width = int(self._beam_kwargs.get("beam_width", 0))
        if beam_width > 0:
            scores, last_tokens, beam_states = self.model.get_initial_beam_state(self.batch_size, beam_width)
            states.update(
                previous_beam_scores=scores,
                previous_beam_last_tokens=last_tokens,
                previous_beam_states=beam_states,
            )
        return states

    def _model_step(self, signals, lengths, states):
        inputs = schemas.PredictInput(inputs=tf.convert_to_tensor(signals), inputs_length=tf.convert_to_tensor(lengths), **states)
        outputs = self.model.recognize_beam(inputs, **self._beam_kwargs) if self._beam_kwargs else self.model.recognize(inputs)

        transcripts = [row.decode("utf-8") for row in self.model.tokenizer.detokenize(outputs.tokens).numpy()]
        next_states = {
            "previous_tokens": outputs.next_tokens,
            "previous_encoder_states": outputs.next_encoder_states,
            "previous_decoder_states": outputs.next_decoder_states,
        }
        if "previous_beam_scores" in states:
            next_states.update(
                previous_beam_scores=outputs.next_beam_scores,
                previous_beam_last_tokens=outputs.next_beam_last_tokens,
                previous_beam_states=outputs.next_beam_states,
            )
        return transcripts, next_states


class ASRInference:
    """
    One session: a single stream of audio, decoded on the shared `ASREngine` for its backend.

    Build one per websocket, or per request. Sessions built from the same `tflite` path or the same
    `model` object share one engine, so they share one loaded model and are batched together.

    A streaming session sends its chunks one at a time and waits for each, because every chunk
    starts from the state the previous one left. Do not call one session from two tasks at once.
    """

    def __init__(
        self,
        tflite: str = None,
        model: BaseModel = None,
        streaming: bool = True,
    ):
        self.streaming = streaming
        self.engine = ASREngine.get(tflite=tflite, model=model, streaming=streaming)
        self.cache = StreamCache()

    def _chunks(self, signal):
        """
        Add `signal` to the buffer and cut out every whole chunk it now completes.

        What is left, including the `size - step` overlap that is the next chunk's left context,
        stays on the cache.
        """
        size, step = self.engine._geometry()
        buffered = signal if self.cache.signal is None else np.concatenate([self.cache.signal, signal])
        chunks = []
        while len(buffered) >= size:
            chunks.append(buffered[:size])
            buffered = buffered[step:]  # step, not size: the overlap is the next chunk's left context
        self.cache = self.cache._replace(signal=buffered)
        return chunks

    def __call__(self, signal) -> str:
        """
        Decode `signal` and wait for the result.

        Non-streaming, this is the transcript of the whole signal. Streaming, it is only what this
        call decoded, and it is empty when the audio so far has not filled a chunk, so a caller
        appends rather than replaces.
        """
        signal = _signal(signal)
        if not self.streaming:
            return self.engine.submit(signal).result()[0]
        transcript = ""
        for chunk in self._chunks(signal):
            text, states = self.engine.submit(chunk, self.cache.states).result()
            self.cache = self.cache._replace(states=states)
            transcript += text
        return transcript

    async def infer(self, signal) -> str:
        """The same as calling the session, but it awaits the engine and does not block the event loop."""
        signal = _signal(signal)
        if not self.streaming:
            return (await asyncio.wrap_future(self.engine.submit(signal)))[0]
        transcript = ""
        for chunk in self._chunks(signal):
            text, states = await asyncio.wrap_future(self.engine.submit(chunk, self.cache.states))
            self.cache = self.cache._replace(states=states)
            transcript += text
        return transcript

    def start(self):
        self.cache = StreamCache()

    def _tail(self):
        """
        The zero padding that completes the last partial chunk, or None if there is nothing to flush.

        The leftover is padded to a whole `signal_chunk_size` because the model only accepts that
        length. It is only worth decoding when there is genuinely new audio in it: after a chunk
        the cache still holds `size - step` samples, but those were already decoded as the tail of
        that chunk and are kept solely as the next one's left context, so a cache no longer than
        that would decode nothing but overlap and padding.
        """
        size, step = self.engine._geometry()
        leftover = 0 if self.cache.signal is None else len(self.cache.signal)
        return np.zeros([size - leftover], dtype=np.float32) if leftover > size - step else None

    def end(self) -> str:
        """Close the session: decode whatever audio is left, then drop the cache."""
        padding = self._tail()
        transcript = "" if padding is None else self(padding)
        self.cache = StreamCache()
        return transcript

    async def aend(self) -> str:
        """The same as `end()`, awaited."""
        padding = self._tail()
        transcript = "" if padding is None else await self.infer(padding)
        self.cache = StreamCache()
        return transcript
