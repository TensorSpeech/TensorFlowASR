- [Inference](#inference)
  - [1. Overview](#1-overview)
  - [2. Building one](#2-building-one)
    - [2.1 From an export](#21-from-an-export)
    - [2.2 From a checkpoint](#22-from-a-checkpoint)
  - [3. One pass](#3-one-pass)
  - [4. Streaming](#4-streaming)
    - [4.1 The session](#41-the-session)
    - [4.2 Why `end()` matters](#42-why-end-matters)
    - [4.3 Block size is latency, not correctness](#43-block-size-is-latency-not-correctness)
    - [4.4 Chunk geometry](#44-chunk-geometry)
  - [5. Many sessions, one model](#5-many-sessions-one-model)
    - [5.1 The shared engine](#51-the-shared-engine)
    - [5.2 How requests are batched](#52-how-requests-are-batched)
    - [5.3 In an async server](#53-in-an-async-server)
  - [6. What streaming costs you](#6-what-streaming-costs-you)
  - [7. Language models](#7-language-models)
  - [8. Examples](#8-examples)
  - [9. Verification](#9-verification)

# Inference

## 1. Overview

[`inferences.py`](../tensorflow_asr/inferences.py) transcribes audio with one of two classes:

| Class          | What it is                                                                         |
| -------------- | ---------------------------------------------------------------------------------- |
| `ASRInference` | one session: one stream of audio, with its own buffer and state                     |
| `ASREngine`    | one loaded model, shared by all sessions, that decodes up to `B` sessions per call  |

You build `ASRInference` objects. Each one gets its engine by itself, and sessions built on the same
backend share that engine. The backend is either of these:

| Built with                  | Decodes with                                          |
| --------------------------- | ----------------------------------------------------- |
| `ASRInference(model=...)`   | a live `BaseModel`, through `recognize` / `recognize_beam` |
| `ASRInference(tflite=...)`  | an exported `.tflite`, through a TFLite interpreter    |

```python
from tensorflow_asr.inferences import ASRInference

asr = ASRInference(tflite="/path/to/model.tflite", streaming=False)
transcript = asr(signal)   # "the transcript"
```

A call takes one signal, `[T]` or `[1, T]`, and returns one `str`. When you build the session, you
choose the mode:

- `streaming=False` decodes the whole signal in one pass.
- `streaming=True` (the default) decodes whatever arrives, a piece at a time.

The session does the work that used to be in every caller. It builds the `schemas.PredictInput`. It
seeds the encoder, decoder and beam states and carries them across calls. It buffers audio that does
not yet fill a chunk, and it turns tokens into text. See [decoders](./decoders.md) for what happens
underneath.

## 2. Building one

### 2.1 From an export

You do not need anything else. `tensorflow_asr tflite` records the sample rate, blank id, beam width
and chunk geometry inside the flatbuffer, so the file describes itself. See
[tflite](./tutorials/tflite.md) for what is stored and why.

```python
asr = ASRInference(tflite="/path/to/model.tflite")
asr.engine.metadata
# {'signal_chunk_size': 2800, 'signal_chunk_step': 2560, 'sample_rate': 16000,
#  'blank': 0, 'beam_width': 0, 'nchunks': 1}
```

An export made before metadata and named inputs existed is refused, with a message that tells you
to export it again. Such a file cannot be run safely, because nothing in it says which new state
replaces which old one.

### 2.2 From a checkpoint

`make()` must run first, and the tokenizer must be attached, because the model turns its own tokens
into text.

```python
import os

from tensorflow_asr import tokenizers
from tensorflow_asr.configs import Config
from tensorflow_asr.inferences import ASRInference
from tensorflow_asr.utils import keras_util

config = Config("/path/to/config.yml.j2", training=False, repodir=os.getcwd())
tokenizer = tokenizers.get(config)
tokenizer.make()

model = keras_util.model_from_config(config.model_config)
model.tokenizer = tokenizer
model.make(batch_size=1)   # the number of sessions decoded per call, see section 5
model.load_weights("/path/to/weights.h5", skip_mismatch=False)

asr = ASRInference(model=model)
```

`repodir` is required, because the config is a jinja template and `repodir` is its include root.

## 3. One pass

```python
asr = ASRInference(tflite="/path/to/model.tflite", streaming=False)
signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav(path, sample_rate=16000))
transcript = asr(signal)
```

The whole signal goes in one call with fresh state, and nothing is kept, so `asr.cache` does not
change. This is exact for every architecture, offline ones too. If the audio is already complete, use
this mode.

## 4. Streaming

### 4.1 The session

`start()` opens a session, each call feeds it more audio, and `end()` closes it. Every call returns
only what it decoded, so the caller appends:

```python
asr = ASRInference(tflite="/path/to/model.tflite")   # streaming=True is the default
asr.start()
transcript = ""
for block in source:                 # microphone, socket, file read in pieces
    transcript += asr(block)
transcript += asr.end()
```

Between calls, the leftover audio and the decoder state stay on `asr.cache`, a `StreamCache`:

| Field    | Holds                                                                        |
| -------- | ---------------------------------------------------------------------------- |
| `signal` | `[n]`: audio not yet decoded, plus the tail that the next chunk overlaps      |
| `states` | this session's slot of the backend state, `None` until something decodes      |

`start()` and `end()` both reset the cache, so an abandoned session cannot leak into the next one.

A session sends its chunks in order and waits for each one, because every chunk starts from the
state that the previous chunk left. Do not call one session from two threads or two tasks at once.

### 4.2 Why `end()` matters

The model only accepts whole chunks. `end()` pads the audio left at the end of a stream with zeros up
to `signal_chunk_size` and decodes it. If you skip `end()`, the last words are never transcribed.

If there is no new audio, it does not pad. After a chunk, the cache still holds
`signal_chunk_size - signal_chunk_step` samples. Those samples were already decoded as the tail of
that chunk, and they stay only as the left context of the next chunk. So a cache no longer than that
decodes nothing, and `end()` returns an empty string.

### 4.3 Block size is latency, not correctness

Feed 100 samples or 100000. Audio that does not fill a chunk waits in the cache, and audio that fills
several chunks decodes several. The transcript is the same either way. Only the wait changes: a block
needs `blocksize / sample_rate` seconds of audio before it can decode.

### 4.4 Chunk geometry

Two numbers control the buffering, and neither one is guessed:

| Number              | Meaning                                    | Comes from                                              |
| ------------------- | ------------------------------------------ | ------------------------------------------------------- |
| `signal_chunk_size` | samples that one model call takes           | `.tflite` metadata, or `get_signal_chunk_size_and_step` |
| `signal_chunk_step` | samples that the buffer moves forward after | the same                                                |

They differ because feature frames overlap. A window is `frame_length` long, but only `frame_step`
new samples arrive per frame, so the tail of one chunk is the head of the next. For this reason the
cache moves forward by `step` and not by `size`.

## 5. Many sessions, one model

### 5.1 The shared engine

`ASRInference` does not load a model. It asks `ASREngine.get(...)` for the engine of its backend and
mode, and the first call builds that engine. So a server with many clients loads the model once.

| Sessions built with                         | Share an engine when                           |
| ------------------------------------------- | ---------------------------------------------- |
| `tflite=path`                               | the path is the same                            |
| `model=model`                               | it is the same model object                     |

Streaming and non-streaming sessions use separate engines. Streaming chunks all have the same
length. Whole signals do not, so they are padded, and padding must never reach a carried state.

For a `.tflite`, each engine loads its own interpreter. For a live model, both engines call the same
model object from their own threads.

### 5.2 How requests are batched

`B` is the batch size of the backend. A model gets it at `make(batch_size=...)`, and an export gets
it at trace time (`tensorflow_asr tflite --bs=B`). It is the most sessions that one call can decode.

Each engine has one worker thread, and it is the only code that runs the model:

1. The worker takes the first request from the queue.
2. It waits up to `max_wait_ms` (5 ms by default) for more requests, until it has `B`.
3. It stacks them into one `[B, width]` call. Empty slots decode silence from the initial state.
4. It gives each session its own transcript and next state.

A non-streaming batch is padded to its longest signal, and each row gets its real length in
`inputs_length`. A row decodes the same text alone or next to other rows. `tests/test_inferences.py`
and `tests/test_inference.py` pin this.

The engine batches chunks from different sessions, never two chunks from the same session.

To set `max_wait_ms`, build the engine before the first session:

```python
from tensorflow_asr.inferences import ASREngine

ASREngine.get(tflite="/path/to/model.tflite", streaming=True, max_wait_ms=10)
```

### 5.3 In an async server

A model call blocks. In an async handler, use `await asr.infer(block)` and `await asr.aend()`, so the
event loop keeps running while the engine decodes. They return the same values as `asr(block)` and
`asr.end()`.

```python
@app.websocket("/asr")
async def asr_socket(websocket: WebSocket):
    await websocket.accept()
    asr = ASRInference(tflite="/path/to/model.tflite")   # one session per websocket
    try:
        while True:
            block = np.frombuffer(await websocket.receive_bytes(), dtype=np.float32)
            await websocket.send_text(await asr.infer(block))
    except WebSocketDisconnect:
        pass
    finally:
        await asr.aend()
```

If the model fails, the error is raised in every session that waits on that call. The worker keeps
running for the next requests.

## 6. What streaming costs you

If the model is causal and the audio is a whole number of chunks, chunked decoding gives
**exactly** the same result as a one-pass decode. `tests/test_inferences.py` pins that in both directions.

In other cases, expect small differences:

- A padded tail. Audio that does not end on a chunk boundary ends with a zero-padded chunk, and those
  zeros can emit a token or two. The one-pass decode is then a strict prefix of the streamed one.
- A non-causal model. You can stream an offline encoder, but it will decode worse, because it never
  sees the future frames that a single pass gives it.
- Beam search. Tokens already emitted cannot change, but a single pass can still change them. See
  [decoders](./decoders.md) 4.10.

A transducer at `B = 1` decodes with `recognize_single`, and at `B > 1` with `recognize_batch`. The
two decoders can give different text for the same audio.

## 7. Language models

You do not need to do anything. On the model backend, the settings come from `decoder_config`
through `get_beam_decoding_kwargs`, the same call that `predict_step` uses. So a beam runs with the
fusion that the config describes, and the beam state is carried across chunks. On an export, the
configuration was fixed at conversion time.

One limit remains: `PredictOutput` has no field for LM state, so a fused LM starts again on every
call. If you feed a fused beam one whole utterance at a time, it is correct. Otherwise it is not. See
[decoders](./decoders.md) 4.10.

## 8. Examples

There are four runnable scripts, one for each way audio can arrive. See
[examples/inferences](../examples/inferences/README.md).

| Script                                                                        | Model      | Audio arrives          |
| ----------------------------------------------------------------------------- | ---------- | ---------------------- |
| [`main.py`](../examples/inferences/main.py)                                   | checkpoint | whole file, one call   |
| [`tflite.py`](../examples/inferences/tflite.py)                               | `.tflite`  | whole file, one call   |
| [`streaming_tflite.py`](../examples/inferences/streaming_tflite.py)           | `.tflite`  | file, block at a time  |
| [`live_streaming_tflite.py`](../examples/inferences/live_streaming_tflite.py) | `.tflite`  | microphone, live       |

## 9. Verification

`tests/test_inferences.py` runs both backends over the same causal streaming Conformer. It pins:

| Property                               | Why it matters                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| Streaming equals one pass              | it needs the geometry, the `step` advance and every state pair to be right at once   |
| State is used                          | each chunk from a fresh session must give a different answer, or streaming is fake    |
| Only the overlap stays                 | keeping `size` decodes audio twice, and keeping nothing drops the shared frames       |
| `end()` flushes only new audio         | decoding pure overlap again adds a phantom repeat to every transcript                 |
| Sessions share one engine              | many clients must not load many models                                                |
| Requests share one call                | `B` requests that arrive together take one model call, and extra requests wait        |
| Batching does not change a transcript  | a session decodes the same alone or with other sessions, and padding does not show    |
| Errors reach the caller                | a failed call raises in every waiting session, and the worker survives                |
| Async equals sync                      | `infer()` and `aend()` return what `__call__` and `end()` return                      |
| Every architecture streams             | all eight export, pair their states and run, whatever the shape of that state         |
