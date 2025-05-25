- [Training Tutorial](#training-tutorial)
  - [1. Install packages](#1-install-packages)
  - [2. Prepare transcripts files](#2-prepare-transcripts-files)
  - [3. Prepare config file](#3-prepare-config-file)
  - [4. \[Optional\]\[Required if using TPUs\] Create tfrecords](#4-optionalrequired-if-using-tpus-create-tfrecords)
  - [5. Generate vocabulary and metadata](#5-generate-vocabulary-and-metadata)
  - [6. Run training](#6-run-training)


# Training Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Installation

```bash
./setup.sh [tpu|gpu|cpu] install
```

## 2. Prepare transcripts files

This is the example for preparing transcript files for librispeech data corpus

```bash
python examples/datasets/librispeech/prepare_transcript.py \
    --directory=/path/to/dataset/train-clean-100 \
    --output=/path/to/dataset/train-clean-100/transcripts.tsv
```

Do the same thing with `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`

For other datasets, please make your own script to prepare the transcript files, take a look at the [`prepare_transcript.py`](../../examples/datasets/librispeech/prepare_transcript.py) file for more reference

## 3. Prepare config file

The config file is under format `config.yml.j2` which is jinja2 format with yaml content

Please take a look in some examples for config files in `examples/*/*.yml.j2`

For example:

```jinja2
{% import "examples/datasets/librispeech/sentencepiece/sp.yml.j2" as decoder_config with context %}
{{decoder_config}}

{% import "examples/models/transducer/conformer/small.yml.j2" as config with context %}
{{config}}
```

## 4. [Optional] Create tfrecords

If you want to train with tfrecords

```bash
tensorflow_asr utils create_tfrecords \
    --config-path=/path/to/config.yml.j2 \
    --mode=\["train","eval","test"\] \
    --datadir=/path/to/datadir
```

You can reduce the flag `--modes` to `--modes=\["train","eval"\]` to only create train and eval datasets

## 5. Generate vocabulary and metadata

This step requires defining path to vocabulary file and other options for generating vocabulary in config file.

```bash
tensorflow_asr utils create_datasets_metadata \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/datadir \
    --dataset-type="slice"
```

The inputs, outputs and other options of vocabulary are defined in the config file

## 6. Run training

```bash
tensorflow_asr train \
    --config-path=/path/to/config.yml.j2 \
    --modeldir=/path/to/modeldir \
    --datadir=/path/to/datadir \
    --dataset-type=tfrecord \ # or "generator" or "slice" \
    --dataset-cache \
    --mxp=strict \
    --bs=4 \
    --ga-steps=8 \
    --verbose=1 \
    --jit-compile \
    --device-type=tpu \
    --tpu-address=local
## See others params
tensorflow_asr train --help
```