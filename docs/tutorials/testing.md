- [Testing Tutorial](#testing-tutorial)
  - [1. Installation](#1-installation)
  - [2. Prepare transcripts files](#2-prepare-transcripts-files)
  - [3. Prepare config file](#3-prepare-config-file)
  - [4. Run testing](#4-run-testing)


# Testing Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Installation

```bash
./setup.sh [tpu|gpu|cpu] install
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
--config-path /path/to/config.yml.j2 \
--dataset_type slice \
--datadir /path/to/datadir \
--outputdir /path/to/modeldir/tests \
--h5 /path/to/modeldir/weights.h5
## See others params
tensorflow_asr test --help
```