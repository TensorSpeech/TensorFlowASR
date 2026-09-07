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
Run a trained model, or an exported `.tflite`, over audio.

A call takes a **batch** of signals, `[B, T]`, and returns one transcript per row. `B` is fixed:
a model bakes it in at `make(batch_size=...)` and an export at trace time, so it is part of the
signature rather than something either can adapt to, and a mismatch is refused. A flat `[T]` vector
is read as a batch of one.

Two modes through that one call:

* **Non-streaming** (`streaming=False`) sends the whole signal in one pass with fresh state and
  keeps nothing. This is exact for every architecture, offline ones included.
* **Streaming** (`streaming=True`) is fed one piece of audio at a time. Samples that do not yet
  fill a chunk stay in `self.cache` along with the decoder state, so the next call continues where
  this one stopped. `start()` opens a session and `end()` closes it -- `end()` zero-pads whatever
  is left to a whole chunk, decodes it, and returns that last piece of transcript.

  A streaming batch advances in lockstep. The rows share one buffer and one set of states, and the
  model consumes the whole batch per invocation, so every row must be handed the same number of
  samples per call. Feeding one row more than another is not representable; pad the short rows.

Both modes run the same per-chunk step. The only difference is where the state and the leftover
samples live: on `self.cache` for streaming, thrown away immediately otherwise.

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

    asr = ASRInference(tflite="/path/to/model.tflite")
    signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav("/path/to/audio.wav", sample_rate=16000))

    print(asr(signal, streaming=False)[0])  # a flat signal is a batch of one, so take row 0

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
    model.make(batch_size=1)
    model.load_weights("/path/to/weights.h5", skip_mismatch=False)

    asr = ASRInference(model=model)
    print(asr(signal, streaming=False)[0])

Streaming, one piece of audio at a time. The pieces have nothing to do with the chunk size -- feed
whatever the source produces and the cache does the aligning. Each call returns only what it
decoded, so the caller appends::

    asr.start()
    transcript = ""
    try:
        for block in microphone():  # any producer: a device, a socket, a file read in pieces
            transcript += asr(block)[0]  # streaming=True is the default
            print(transcript, end="\r", flush=True)
    except KeyboardInterrupt:
        pass  # the escape hatch; `end()` still runs below
    transcript += asr.end()[0]  # pads the last partial chunk, decodes it, closes the session
    print(transcript)

Several streams at once, on a model or export built for that batch. Rows must be equal length, and
padding one to match the others is the caller's job::

    asr = ASRInference(tflite="/path/to/model.tflite")  # exported with --bs=4
    width = max(len(signal) for signal in signals)
    batch = np.stack([np.pad(signal, [0, width - len(signal)]) for signal in signals])  # [4, width]

    for row, transcript in enumerate(asr(batch, streaming=False)):
        print(row, transcript)

To stream a batch, hand every row the same number of samples per call -- one `[B, n]` array per
call, not one row at a time.
"""

import logging
import re
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


def _signals(signals) -> np.ndarray:
    """
    Audio as a `[B, T]` float32 array, whatever shape or container it arrived in.

    A flat vector is a batch of one, which is what an interactive caller with a single microphone
    has. Anything of higher rank is rejected by the caller's batch check rather than silently
    reshaped, since there is no reading of `[B, T, C]` that is obviously right.
    """
    return np.atleast_2d(np.asarray(signals, dtype=np.float32))


class StreamCache(typing.NamedTuple):
    """
    What a streaming session carries from one call to the next, on `ASRInference.cache`.

    `signal` is a `[B, n]` buffer: audio that has arrived but not yet filled a chunk, plus the tail
    of the last chunk that the next one overlaps. All rows hold the same `n`, since the batch is
    decoded together and advances together. `states` is whatever the backend threads through -- a
    list of arrays for a `.tflite`, the trailing fields of `schemas.PredictInput` for a live model
    -- and is deliberately opaque, so one loop drives both.

    Both are `None` until the session has something to remember: no audio buffered, and no state
    because nothing has decoded yet.

    Being a tuple it is replaced rather than edited, so `start()` and `end()` reset by assigning an
    empty one.
    """

    signal: typing.Optional[np.ndarray] = None
    states: typing.Any = None


class ASRInference:
    def __init__(
        self,
        tflite: str = None,
        model: BaseModel = None,
    ):
        if not tflite and not model:
            raise ValueError("Either `tflite` or `model` must be provided.")
        self.tflite = tflite
        self.model = model
        self.cache = StreamCache()
        # Loading the interpreter and reading the metadata is done once, here, rather than on the
        # first chunk: a missing file or an export too old to carry metadata should fail when the
        # object is built, not part way through someone's audio stream.
        self._setup_tflite() if tflite else self._setup_model()

    def _as_batch(self, signals) -> np.ndarray:
        """
        Audio as `[B, T]`, checked against the batch this backend was built for.

        Checked here so the message names the mismatch. Left to the interpreter it surfaces as a
        resize failure, and on the model side a wrong batch reaches the decoder as a shape error
        from somewhere inside a `tf.while_loop`.
        """
        signals = _signals(signals)
        if signals.ndim != 2:
            raise ValueError(f"signals must be [batch, samples], got shape {signals.shape}")
        if signals.shape[0] != self.batch_size:
            built = "export" if self.tflite else "model"
            raise ValueError(f"this {built} takes {self.batch_size} signal(s) per call, got {signals.shape[0]}")
        return signals

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

    def _tflite_step(self, signals, states):
        """One invocation on one chunk of the batch. Returns a transcript per row, and the next state."""
        if self._allocated != signals.shape[1]:
            self.interpreter.resize_tensor_input(self._signal["index"], list(signals.shape), strict=True)
            self.interpreter.allocate_tensors()
            self._allocated = signals.shape[1]
            self._locate_tensors()  # allocation rebuilds the descriptors, tensor indices included

        self.interpreter.set_tensor(self._signal["index"], signals)
        self.interpreter.set_tensor(self._length["index"], np.full([self.batch_size], signals.shape[1], dtype=self._length["dtype"]))
        for detail, value in zip(self._states, states):
            self.interpreter.set_tensor(detail["index"], value)

        self.interpreter.invoke()

        transcripts = [row.decode("utf-8") for row in self.interpreter.get_tensor(self._outputs[0]["index"])]
        # `get_tensor` copies, so the carried state survives the next call overwriting the buffers.
        next_states = list(states)
        for position, tensor_index in self._feedback:
            next_states[position] = self.interpreter.get_tensor(tensor_index)
        return transcripts, next_states

    def _tflite_inference(self, signals, streaming=True):
        signals = self._as_batch(signals)
        if not streaming:
            transcripts, _ = self._tflite_step(signals, self._tflite_initial_states())
            return transcripts
        return self._consume(signals, self._tflite_step, self._tflite_initial_states)

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

    def _model_step(self, signals, states):
        signals = tf.convert_to_tensor(signals)
        # Every row is the same length, so one width fills the whole `[B]` vector.
        inputs_length = tf.fill([self.batch_size], tf.shape(signals)[1])
        inputs = schemas.PredictInput(inputs=signals, inputs_length=inputs_length, **states)
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

    def _model_inference(self, signals, streaming=True):
        signals = self._as_batch(signals)
        if not streaming:
            transcripts, _ = self._model_step(signals, self._model_initial_states())
            return transcripts
        return self._consume(signals, self._model_step, self._model_initial_states)

    # ----------------------------------- STREAMING ----------------------------------- #

    def _geometry(self):
        """`(signal_chunk_size, signal_chunk_step)` -- samples per call, and samples per advance."""
        if self.tflite:
            return int(self.metadata["signal_chunk_size"]), int(self.metadata["signal_chunk_step"])
        size, step = self.model.get_signal_chunk_size_and_step(1)
        return int(size), int(step)

    def _consume(self, signals, step_fn, initial_states):
        """
        Add `signals` to the cache and decode every whole chunk the batch now completes.

        Returns one transcript per row, holding only what this call decoded, so a caller appends
        rather than replaces. Every row is empty whenever the audio so far has not filled a chunk.
        """
        size, step = self._geometry()
        buffered = self.cache.signal
        signal = signals if buffered is None else np.concatenate([buffered, signals], axis=1)
        states = self.cache.states

        decoded = [[] for _ in range(self.batch_size)]
        while signal.shape[1] >= size:
            # Seeded here rather than above so that a call too short to fill a chunk only buffers
            # audio, leaving `states` None as `StreamCache` documents. Building the initial state
            # costs a handful of allocations, and a caller feeding small pieces makes many such
            # calls before the first chunk is ready.
            states = initial_states() if states is None else states
            transcripts, states = step_fn(signal[:, :size], states)
            for row, transcript in enumerate(transcripts):
                decoded[row].append(transcript)
            signal = signal[:, step:]  # step, not size: the overlap is the next chunk's left context

        self.cache = StreamCache(signal=signal, states=states)
        return ["".join(parts) for parts in decoded]

    def start(self):
        self.cache = StreamCache()

    def __call__(self, signals, streaming=True):
        if self.tflite:
            return self._tflite_inference(signals, streaming)
        elif self.model:
            return self._model_inference(signals, streaming)
        else:
            raise ValueError("Either `tflite` or `model` must be provided.")

    def end(self):
        """
        Close the session: decode whatever audio is left, then drop the cache.

        The leftover is padded with zeros to a whole `signal_chunk_size` because the model only
        accepts that length. It is only worth decoding when there is genuinely new audio in it:
        after a chunk the cache still holds `size - step` samples, but those were already decoded
        as the tail of that chunk and are kept solely as the next one's left context, so a cache no
        longer than that would decode nothing but overlap and padding.
        """
        transcripts = [""] * self.batch_size
        size, step = self._geometry()
        leftover = 0 if self.cache.signal is None else self.cache.signal.shape[1]
        if leftover > size - step:
            transcripts = self(np.zeros([self.batch_size, size - leftover], dtype=np.float32), streaming=True)
        self.cache = StreamCache()
        return transcripts
