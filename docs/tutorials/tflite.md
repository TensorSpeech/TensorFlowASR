- [TFLite Tutorial](#tflite-tutorial)
  - [Conversion](#conversion)
  - [Inference](#inference)
    - [1. Input](#1-input)
    - [2. Output](#2-output)
    - [3. Example script](#3-example-script)


# TFLite Tutorial

## Conversion

```bash
python3 examples/tflite.py \
    --config-path=/path/to/config.yml.j2 \
    --h5=/path/to/weight.h5 \
    --bs=1 \ # Batch size
    --beam-width=0 \ # Beam width, set >0 to enable beam search
    --output=/path/to/output.tflite
## See others params
python examples/tflite.py --help
```

## Inference

### 1. Input

Input of each tflite depends on the models' parameters and configs.

The `inputs`, `inputs_length` and `previous_tokens` are still the same as bellow for all models.

```python
schemas.PredictInput(
    inputs=tf.TensorSpec([batch_size, None], dtype=tf.float32),
    inputs_length=tf.TensorSpec([batch_size], dtype=tf.int32),
    previous_tokens=tf.TensorSpec.from_tensor(self.get_initial_tokens(batch_size)),
    previous_encoder_states=tf.TensorSpec.from_tensor(self.get_initial_encoder_states(batch_size)),
    previous_decoder_states=tf.TensorSpec.from_tensor(self.get_initial_decoder_states(batch_size)),
)
```

For models that don't have encoder states or decoder states, the default values are `tf.zeros([], dtype=self.dtype)` tensors for `previous_encoder_states` and `previous_decoder_states`. This is just for tflite conversion because tflite does not allow `None` value in `input_signature`. However, the output `next_encoder_states` and `next_decoder_states` are still `None`, so we can simply ignore those outputs.

### 2. Output

```python
schemas.PredictOutputWithTranscript(
    transcript=self.tokenizer.detokenize(outputs.tokens),
    tokens=outputs.tokens,
    next_tokens=outputs.next_tokens,
    next_encoder_states=outputs.next_encoder_states,
    next_decoder_states=outputs.next_decoder_states,
)
```

This is for supporting streaming inference.

Each output corresponds to the input = each chunk of audio signal.

Then we can overwrite `previous_tokens`, `previous_encoder_states` and `previous_decoder_states` with `next_tokens`, `next_encoder_states` and `next_decoder_states` for the next chunk of audio signal.

And continue until the end of the audio signal.

### 3. Example script

See [examples/inferences/tflite.py](../../examples/inferences/tflite.py) for more details.