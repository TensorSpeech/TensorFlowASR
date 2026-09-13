- [Testing Tutorial](#testing-tutorial)
  - [1. Installation](#1-installation)
  - [2. Prepare transcripts files](#2-prepare-transcripts-files)
  - [3. Prepare config file](#3-prepare-config-file)
  - [4. Run testing](#4-run-testing)
  - [5. Decoding with a language model](#5-decoding-with-a-language-model)


# Testing Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Installation

```bash
uv sync                 # CPU / Apple Silicon
uv sync --extra cuda    # NVIDIA GPU

# TPU: sync first, then swap TensorFlow for the Cloud TPU build. This cannot be an
# extra -- `tensorflow-tpu` ships its own `tensorflow` distribution, so the two
# cannot be installed together. Re-run it after any later `uv sync`.
uv sync && ./scripts/install_tpu.sh
```

## 2. Prepare transcripts files

This is the example for preparing transcript files for librispeech data corpus

```bash
python examples/datasets/librispeech/prepare_transcript.py \
    --directory=/path/to/dataset/test-clean \
    --output=/path/to/dataset/test-clean/transcripts.tsv
```

Do the same thing with `test-clean`, `test-other`

For other datasets, please make your own script to prepare the transcript files, take a look at the [`prepare_transcript.py`](../../examples/datasets/librispeech/prepare_transcript.py) file for more reference

## 3. Prepare config file

The config file is under format `config.yml.j2` which is jinja2 format with yaml content

Please take a look in some examples for config files in `examples/*/*.yml.j2`

The config file is the same as the config used for training

The inputs, outputs and other options of vocabulary are defined in the config file

For example:

```jinja2
{% import "examples/datasets/librispeech/sentencepiece/sp.yml.j2" as decoder_config with context %}
{{decoder_config}}

{% import "examples/models/transducer/conformer/small.yml.j2" as config with context %}
{{config}}
```

## 4. Run testing

```bash
tensorflow_asr test \
    --config-path=/path/to/config.yml.j2 \
    --dataset-type=slice \
    --datadir=/path/to/datadir \
    --outputdir=/path/to/modeldir/tests \
    --h5=/path/to/modeldir/weights.h5
## See others params
tensorflow_asr test --help
```

`--dataset-type` is one of `tfrecord`, `slice`, `generator` or `huggingface`, and must match how the
data was prepared in step 2.

## 5. Decoding with a language model

Beam search and language-model fusion are configured in `decoder_config` — `beam_width`, `lm_type`,
`lm_alpha`, `lm_beta` — and the trained weights are passed on the command line, because which copy
you score with is a property of the run rather than of the config:

```bash
tensorflow_asr test \
    --config-path=/path/to/config.yml.j2 \
    --dataset-type=slice \
    --datadir=/path/to/datadir \
    --outputdir=/path/to/modeldir/tests \
    --h5=/path/to/modeldir/weights.h5 \
    --lm-h5=/path/to/modeldir/lm/external.weights.h5 \
    --internal-lm-h5=/path/to/modeldir/lm/internal.weights.h5 \
    --lm-type=lodr
```

`--internal-lm-h5` is only read when `lm_type` is `"lodr"`. `--lm-type` overrides
`decoder_config.lm_type` for one run, so the three corrections can be compared over one checkpoint
without editing the config. The report carries a plain beam and a fused beam side by side, so you
can read off what the language model changed. See [decoders](../decoders.md) for what each mode
does, and [training](./training.md) 7 for how the weights are produced.