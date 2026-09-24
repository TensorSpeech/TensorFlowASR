- [TFLite Tutorial](#tflite-tutorial)
  - [1. Conversion](#1-conversion)
    - [1.1 Batch size is baked in](#11-batch-size-is-baked-in)
    - [1.2 Beam search and language models](#12-beam-search-and-language-models)
    - [1.3 The Flex delegate](#13-the-flex-delegate)
  - [2. What the file carries](#2-what-the-file-carries)
    - [2.1 Metadata](#21-metadata)
    - [2.2 Signature](#22-signature)
    - [2.3 Why the inputs are named](#23-why-the-inputs-are-named)
  - [3. Inference](#3-inference)

# TFLite Tutorial

## 1. Conversion

```bash
tensorflow_asr tflite \
    --config-path=/path/to/config.yml.j2 \
    --h5=/path/to/weight.h5 \
    --bs=1 \ # Batch size
    --beam-width=0 \ # Beam width, set >0 to enable beam search
    --nchunks=1 \ # Attention chunks the recorded chunk geometry covers
    --output=/path/to/output.tflite
## See others params
tensorflow_asr tflite --help
```

`--nchunks` changes nothing about the exported graph — only the chunk geometry written into the
metadata, which is a streaming client's latency knob. Larger means fewer, bigger calls for an
unchanged transcript. Both lengths are linear in it, so a client can recover a different `n` from
the recorded pair without re-exporting.

### 1.1 Batch size is baked in

`--bs` fixes the leading dimension of every input at trace time. It is part of the signature, not
something the file adapts to later: an export made at 4 takes four signals per call and refuses one.
Export at `--bs=1` unless you are batching deliberately.

### 1.2 Beam search and language models

`--beam-width` above 0 exports the ALSD++ beam search instead of the greedy decoder, together with
the language model settings from `decoder_config` — fusion weight, correction type, and the models
themselves, frozen into the flatbuffer beside the ASR weights. It overrides
`decoder_config.beam_width` rather than reading it, since the shipped configs leave that at 0.

Pass `--lm-h5` (and `--internal-lm-h5` when `lm_type` is `"lodr"`) so the language models are frozen
in with their trained weights. Without them the model `lm_config` describes is exported with its
*initial* weights and contributes noise; `tensorflow_asr tflite` warns about that and the other
silent misconfigurations through the same `validate_lm` that `tensorflow_asr test` uses.

```bash
tensorflow_asr tflite \
    --config-path=/path/to/config.yml.j2 \
    --h5=/path/to/weight.h5 \
    --lm-h5=/path/to/lm.h5 \
    --internal-lm-h5=/path/to/ilm.h5 \
    --bs=1 \
    --beam-width=16 \
    --output=/path/to/output.tflite
```

A fused export restarts the LM on every call, so it is only correct fed one whole utterance at a
time — see [decoders](../decoders.md) 4.8 and 4.10.

### 1.3 The Flex delegate

These models need `SELECT_TF_OPS`: the decoders' `tf.while_loop` and the in-graph detokenization
have no TFLite builtin equivalents. **TensorFlow 2.20 dropped the Flex delegate from the pip
wheel**, so conversion still works there but inference fails with:

```
RuntimeError: Select TensorFlow op(s), included in the given model, is(are) not
supported by this interpreter. Make sure you apply/link the Flex delegate before inference.
```

A flatbuffer produced by 2.20 loads fine in a 2.18/2.19 interpreter, so the two halves can be split
across environments. On Android the delegate is a dependency:
`org.tensorflow:tensorflow-lite-select-tf-ops`.

Convert with **no GPU visible**. Keras picks the fused LSTM kernel whenever one is *visible* —
placement is irrelevant — and that kernel converts to a `CudnnRNNV3` custom op no interpreter can
resolve. See the note on `app_util.convert_tflite`.

## 2. What the file carries

### 2.1 Metadata

Conversion stores a JSON blob under `TFASR_METADATA` in the flatbuffer's own `metadata` field, so a
deployed model needs neither a sidecar file nor the Python config that produced it:

```python
from tensorflow_asr.utils import tflite_util

tflite_util.read_metadata("/path/to/model.tflite")
# {'signal_chunk_size': 2800, 'signal_chunk_step': 2560, 'sample_rate': 16000,
#  'blank': 0, 'beam_width': 0, 'nchunks': 1}
```

| Key                 | What a client does with it                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| `signal_chunk_size` | samples to pass as `inputs` in one streaming call                              |
| `signal_chunk_step` | samples to advance by afterwards — smaller than the size, because frames overlap |
| `sample_rate`       | the rate to resample incoming audio to                                          |
| `blank`             | the token id to seed `previous_tokens` with                                     |
| `beam_width`        | the width traced, so 0 means greedy and the `previous_beam_*` inputs are absent  |
| `nchunks`           | attention chunks the recorded geometry covers, so the two lengths can be read    |

The interpreter never reads any of it, so it costs nothing at inference. Reading it back is not part
of `tf.lite.Interpreter`'s API — a non-Python client writes the equivalent of `read_metadata`
against the same schema. See [tflite_util.py](../../tensorflow_asr/utils/tflite_util.py).

### 2.2 Signature

The signature is `schemas.PredictInput` in, `schemas.PredictOutputWithTranscript` out, both
flattened. `previous_encoder_states` and `previous_decoder_states` are nested structures rather than
single tensors, so each contributes as many inputs as it has leaves:

```python
input_signature = schemas.PredictInput(
    inputs=tf.TensorSpec([batch_size, None], dtype=tf.float32),
    inputs_length=tf.TensorSpec([batch_size], dtype=tf.int32),
    previous_tokens=tf.TensorSpec.from_tensor(self.get_initial_tokens(batch_size)),
    previous_encoder_states=tf.nest.map_structure(tf.TensorSpec.from_tensor, self.get_initial_encoder_states(batch_size)),
    previous_decoder_states=tf.nest.map_structure(tf.TensorSpec.from_tensor, self.get_initial_decoder_states(batch_size)),
    **beam_signature,   # previous_beam_scores / _last_tokens / _states, only when beam_width > 0
)
```

The time axis of `inputs` is the only dynamic dimension; everything else is static. A model with no
encoder or decoder state contributes no inputs for it — `[]` flattens to nothing — and a model that
returns `None` for an output drops it from the file entirely, which is why a CTC export has one more
state input than it has state outputs (`next_tokens` is None for a non-autoregressive decoder).

Outputs are numbered in flattened order: `Identity`, `Identity_1`, … before the variables are
frozen, `StatefulPartitionedCall:N` or `PartitionedCall:N` after. The transcript is output 0,
produced inside the graph, so no tokenizer is needed on the client side.

### 2.3 Why the inputs are named

Every input spec is named `tfasr_input_<position>`, after its position in the flattened signature.

This is not cosmetic. An unnamed `tf.TensorSpec` leaves `tf.function` to label the placeholders
`inputs`, `inputs_1`, … in an order of its own — a traced Conformer puts leaf 3 in `inputs_5` — and
the name is the only thing the flatbuffer keeps. Since outputs *are* numbered in flattened order,
an export whose inputs are auto-named cannot be streamed: nothing in the file says which new state
replaces which old one, and pairing them by position feeds a convolution cache into a subsampling
slot without raising.

`get_input_details()` returns tensors in interpreter order, not signature order, and
`get_signature_list()` is empty for some architectures — so sort on the name.

An export made before this existed is refused by `ASRInference`, with a message saying to re-export.
Older files still work for a one-pass decode, where no state is fed back.

## 3. Inference

Use [`ASRInference`](../inferences.md), which locates the tensors, seeds the carried state, feeds
each new state back into the input it belongs to, and decodes the transcript bytes:

```python
from tensorflow_asr.inferences import ASRInference

asr = ASRInference(tflite="/path/to/model.tflite", streaming=False)
transcript = asr(signal)
```

For streaming, build the session with `streaming=True` (the default), call `asr.start()`, call it with
each block as it arrives, then call `asr.end()` to flush the padded tail. Sessions on the same file
share one interpreter, which decodes up to `--bs` sessions per call. The full contract is in
[inferences](../inferences.md): sessions, the shared engine, the cache, chunk geometry and what
streaming costs.

Runnable scripts for both, plus a microphone, are in
[examples/inferences](../../examples/inferences/README.md).
