- [Testing Tutorial](#testing-tutorial)
  - [1. Install packages](#1-install-packages)
  - [2. Prepare transcripts files](#2-prepare-transcripts-files)
  - [3. Prepare config file](#3-prepare-config-file)
  - [4. \[Optional\]\[Required if not exists\] Generate vocabulary and metadata](#4-optionalrequired-if-not-exists-generate-vocabulary-and-metadata)
  - [5. Run testing](#5-run-testing)


# Testing Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Install packages

If you use google colab, it's recommended to use the tensorflow version pre-installed on the colab itself

```bash
pip uninstall -y TensorFlowASR # uninstall for clean install if needed
pip install ".[tf2.x]"
```

## 2. Prepare transcripts files

This is the example for preparing transcript files for librispeech data corpus

```bash
tensorflow_asr utils create_librispeech_trans \
    --directory=/path/to/dataset/test-clean \
    --output=/path/to/dataset/test-clean/transcripts.tsv
```

Do the same thing with `test-clean`, `test-other`

For other datasets, you must prepare your own python script like the `tensorflow_asr/scripts/utils/create_librispeech_trans.py`

## 3. Prepare config file

The config file is under format `config.yml.j2` which is jinja2 format with yaml content

Please take a look in some examples for config files in `examples/*/*.yml.j2`

The config file is the same as the config used for training

## 4. [Optional][Required if not exists] Generate vocabulary and metadata

Use the same vocabulary file used in training

```bash
tensorflow_asr utils prepare_vocab_and_metadata \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/datadir
```

The inputs, outputs and other options of vocabulary are defined in the config file

## 5. Run testing

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