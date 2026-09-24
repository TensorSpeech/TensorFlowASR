# Inference Examples

Four ways to transcribe with a trained model, from the smallest thing that works to a live
microphone. All four decode through [`ASRInference`](../../tensorflow_asr/inferences.py), so the
difference between them is only where the audio comes from and whether it arrives all at once.

| Example | Model | Audio arrives | Use it to |
| --- | --- | --- | --- |
| [`main.py`](./main.py) | checkpoint (`.h5` + config) | whole file, one call | transcribe with weights you just trained, before exporting anything |
| [`tflite.py`](./tflite.py) | exported `.tflite` | whole file, one call | check an export transcribes what the checkpoint did |
| [`streaming_tflite.py`](./streaming_tflite.py) | exported `.tflite` | file, one block at a time | develop and debug the streaming path deterministically |
| [`live_streaming_tflite.py`](./live_streaming_tflite.py) | exported `.tflite` | microphone, live | transcribe as you speak |

Each file's docstring covers the details; the flags below are the whole interface.

## From a checkpoint

Needs the config and the tokenizer, because a checkpoint carries neither.

```bash
python examples/inferences/main.py \
    --file-path ./examples/inferences/wavs/1089-134691-0000.flac \
    --config-path /path/to/config.yml.j2 \
    --h5 /path/to/weights.h5
```

## From an export, one pass

Needs nothing but the file. `tensorflow_asr tflite` records the sample rate, the blank id and the
chunk geometry in the flatbuffer's own metadata, and the transcript is produced inside the graph,
so there is no config to load, no tokenizer to build and no sample rate to pass — one wrong number
there would transcribe nonsense rather than fail.

```bash
python examples/inferences/tflite.py \
    --audio-file-path ./examples/inferences/wavs/1089-134691-0000.flac \
    --tflite /path/to/model.tflite
```

## From an export, block by block

The same loop a server or a phone would run, with a file standing in for the live source: it only
ever sees the next block, never the whole signal.

```bash
python examples/inferences/streaming_tflite.py \
    --audio-file-path ./examples/inferences/wavs/1089-134691-0000.flac \
    --tflite /path/to/model.tflite \
    --blocksize 2560   # optional; defaults to one signal_chunk_step
```

`--blocksize` is latency, not correctness. Feed 100 samples or 100000 — `ASRInference` holds back
whatever does not yet fill a chunk, so the transcript is the same either way.

## From the microphone

`sounddevice` is already a dependency of this package, so there is nothing extra to install. Press
Ctrl-C to stop; the last partial chunk is padded and decoded on the way out, so the tail of what
you said is not lost.

```bash
# list the recording devices, if the default is not the one you want
python -c "import sounddevice; print(sounddevice.query_devices())"

python examples/inferences/live_streaming_tflite.py \
    --tflite /path/to/model.tflite \
    --device 2   # optional; id or name, default device when unset
```

## Two things that trip people up

**The export batch size is sessions per call.** `tensorflow_asr tflite --bs=B` sets how many sessions
one call can decode. Each script here runs one session, so any `B` works, and the other slots decode
silence. `--bs=1` is the cheapest for a single session. A server with many clients can use a larger
`B`, see [inferences](../../docs/inferences.md) section 5.

**A streamed transcript may end slightly longer than a one-pass one.** The last chunk is zero-padded
up to a whole `signal_chunk_size` before it is decoded, since that is the only length the model
accepts, and those zeros can emit a token or two. On a causal model with chunk-aligned audio the two
agree exactly — `tests/test_inferences.py` pins that — and otherwise the one-pass decode is a prefix
of the streamed one.

## Wave files

`wavs/` holds two utterances from LibriSpeech test-clean and test-other.
