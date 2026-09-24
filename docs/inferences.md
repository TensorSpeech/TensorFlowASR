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
  - [5. Batches](#5-batches)
  - [6. What streaming costs you](#6-what-streaming-costs-you)
  - [7. Language models](#7-language-models)
  - [8. Examples](#8-examples)
  - [9. Verification](#9-verification)

# Inference

## 1. Overview

[`ASRInference`](../tensorflow_asr/inferences.py) is the entry point for transcribing audio. It
wraps either backend behind one call:

| Built with                | Decodes with                                             |
| ------------------------- | -------------------------------------------------------- |
| `ASRInference(model=...)` | a live `BaseModel` — `recognize` / `recognize_beam`        |
| `ASRInference(tflite=...)` | an exported `.tflite`, through a TFLite interpreter       |

```python
from tensorflow_asr.inferences import ASRInference

asr = ASRInference(tflite="/path/to/model.tflite")
transcripts = asr(signal, streaming=False)   # ["the transcript"]
```

A call takes a **batch** of signals, `[B, T]`, and returns one transcript per row — always a list,
even at `B = 1`. A flat `[T]` vector is read as a batch of one. `streaming=False` decodes the whole
signal in one pass; `streaming=True` decodes whatever arrives, a piece at a time.

What it does for you is the bookkeeping that used to sit in every caller: building the
`schemas.PredictInput`, seeding and threading the encoder, decoder and beam states across calls,
buffering audio that does not yet fill a chunk, and turning tokens into text. See
[decoders](./decoders.md) for what happens underneath.

## 2. Building one

### 2.1 From an export

Nothing else is needed. `tensorflow_asr tflite` records the sample rate, blank id, beam width and
chunk geometry inside the flatbuffer, so the file describes itself — see
[tflite](./tutorials/tflite.md) for what is stored and why.

```python
asr = ASRInference(tflite="/path/to/model.tflite")
asr.metadata
# {'signal_chunk_size': 2800, 'signal_chunk_step': 2560, 'sample_rate': 16000,
#  'blank': 0, 'beam_width': 0, 'nchunks': 1}
```

An export made before metadata and named inputs existed is refused, with a message saying to
re-export. It cannot be driven safely: nothing in such a file says which new state replaces which
old one.

### 2.2 From a checkpoint

`make()` must have run and the tokenizer must be attached — the model detokenizes its own output.

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
model.make(batch_size=1)
model.load_weights("/path/to/weights.h5", skip_mismatch=False)

asr = ASRInference(model=model)
```

`repodir` is not optional: the config is a jinja template and that is its include root.

## 3. One pass

```python
signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav(path, sample_rate=16000))
transcript = asr(signal, streaming=False)[0]
```

The whole signal goes in one call with fresh state, and nothing is kept — `asr.cache` is untouched.
This is exact for every architecture, offline ones included, and is what you want whenever the
audio is already complete.

## 4. Streaming

### 4.1 The session

`start()` opens a session, each call feeds it more audio, `end()` closes it. Every call returns
only what it decoded, so the caller appends:

```python
asr.start()
transcript = ""
for block in source:                 # microphone, socket, file read in pieces
    transcript += asr(block)[0]      # streaming=True is the default
transcript += asr.end()[0]
```

Between calls the leftover audio and the decoder state live on `asr.cache`, a `StreamCache`:

| Field    | Holds                                                                             |
| -------- | --------------------------------------------------------------------------------- |
| `signal` | `[B, n]` — audio not yet decoded, plus the tail the next chunk overlaps            |
| `states` | whatever the backend threads through; `None` until something has actually decoded  |

`start()` and `end()` both reset it, so an abandoned session cannot leak into the next one.

### 4.2 Why `end()` matters

The model only accepts whole chunks, so audio left over at the end of a stream is zero-padded up to
`signal_chunk_size` and decoded by `end()`. Skip it and the last words are never transcribed.

It pads only when there is genuinely new audio. After a chunk the cache still holds
`signal_chunk_size - signal_chunk_step` samples, but those were already decoded as that chunk's
tail and are kept solely as the next one's left context — so a cache no larger than that decodes
nothing, and `end()` returns empty strings.

### 4.3 Block size is latency, not correctness

Feed 100 samples or 100000. Anything that does not fill a chunk waits in the cache, and anything
that fills several decodes several. The transcript is the same either way; what changes is how long
you wait for it — `blocksize / sample_rate` seconds before a block can be decoded at all.

### 4.4 Chunk geometry

Two numbers drive the buffering, and neither is ever guessed:

| Number              | Meaning                                    | Comes from                                        |
| ------------------- | ------------------------------------------ | ------------------------------------------------- |
| `signal_chunk_size` | samples a call to the model consumes        | `.tflite` metadata, or `get_signal_chunk_size_and_step` |
| `signal_chunk_step` | samples the buffer advances afterwards      | the same                                          |

They differ because consecutive feature frames overlap: a window is `frame_length` long but only
`frame_step` new samples arrive per frame, so the tail of one chunk is the head of the next. That is
why the cache advances by `step` and not by `size`.

## 5. Batches

`B` is fixed. A model bakes it in at `make(batch_size=...)`, an export at trace time, so it is part
of the signature rather than something either can adapt to. A mismatch is refused by name:

```
ValueError: this export takes 2 signal(s) per call, got 1
```

A streaming batch advances in **lockstep**. The rows share one buffer and one set of states, and
the model consumes the whole batch per invocation, so every row must be handed the same number of
samples per call. Feeding one row more than another is not representable — pad the short rows.

## 6. What streaming costs you

Chunked decoding reproduces a one-pass decode **exactly** when the model is causal and the audio is
a whole number of chunks. `tests/test_inferences.py` pins that in both directions.

Off that path, expect small differences, and know which:

- **A padded tail.** Audio that is not chunk-aligned ends with a zero-padded final chunk, and those
  zeros can emit a token or two. The one-pass decode is then a strict prefix of the streamed one.
- **A non-causal model.** Streaming an offline encoder is legal and will simply decode worse: it
  never sees the future frames a single pass would have given it.
- **Beam search stays approximate.** Tokens already emitted cannot be revised, while a single pass
  still can. See [decoders](./decoders.md) 4.10.

## 7. Language models

Nothing extra to do. On the model backend the settings come from `decoder_config` through
`get_beam_decoding_kwargs`, the same call `predict_step` uses, so a beam runs with whatever fusion
the config describes and the beam state is carried across chunks. On an export they were frozen in
at conversion time.

One limitation carries over: `PredictOutput` has no field for LM state, so a fused LM restarts on
every call. A fused beam is therefore only correct fed one whole utterance at a time — see
[decoders](./decoders.md) 4.10.

## 8. Examples

Four runnable scripts, one per way audio can arrive. See
[examples/inferences](../examples/inferences/README.md).

| Script                                                                          | Model      | Audio arrives          |
| ------------------------------------------------------------------------------- | ---------- | ---------------------- |
| [`main.py`](../examples/inferences/main.py)                                     | checkpoint | whole file, one call   |
| [`tflite.py`](../examples/inferences/tflite.py)                                 | `.tflite`  | whole file, one call   |
| [`streaming_tflite.py`](../examples/inferences/streaming_tflite.py)             | `.tflite`  | file, block at a time  |
| [`live_streaming_tflite.py`](../examples/inferences/live_streaming_tflite.py)   | `.tflite`  | microphone, live       |

## 9. Verification

`tests/test_inferences.py` drives both backends over the same causal streaming Conformer. What it
pins:

| Property                        | Why it matters                                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| Streaming equals one pass       | only holds if geometry, the `step` advance and every state feedback pair are right at once       |
| State is actually consumed      | re-running each chunk from a fresh session must give a different answer, or "streaming" is fiction |
| Only the overlap is retained    | keeping `size` would decode audio twice; keeping nothing would drop the frames chunks share      |
| `end()` flushes, but only new audio | a padded re-decode of pure overlap would append a phantom repeat to every transcript         |
| Every architecture streams      | all eight export, pair their states and run, whatever the shape of that state                    |
| Rows are independent            | one row's audio must not change another's transcript                                             |
| A wrong batch is refused        | named at the call rather than failing inside the interpreter                                     |
