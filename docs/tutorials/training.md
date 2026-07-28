- [Training Tutorial](#training-tutorial)
  - [1. Installation](#1-installation)
  - [2. Prepare transcripts files](#2-prepare-transcripts-files)
  - [3. Prepare config file](#3-prepare-config-file)
  - [4. \[Optional\] Create tfrecords](#4-optional-create-tfrecords)
  - [5. Generate vocabulary and metadata](#5-generate-vocabulary-and-metadata)
  - [6. Run training](#6-run-training)
  - [7. \[Optional\] Train a language model](#7-optional-train-a-language-model)


# Training Tutorial

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

## 7. [Optional] Train a language model

Only needed if you decode with beam search and a language model. Greedy decoding uses none, so you can stop at step 6.

There are two language models, and `--target` picks which one is trained:

| `--target`   | What it is                                        | Trained on                       | Config key                 |
| ------------ | ------------------------------------------------- | -------------------------------- | -------------------------- |
| `external`   | the LM fused *into* the scores                    | a large text corpus              | `lm_config.external_config` |
| `internal`   | the low-order LM that LODR *subtracts*            | the ASR training transcripts     | `lm_config.internal_config` |

Which ones you need depends on `decoder_config.lm_type`: `shallow` and `ilme` use the external LM only, `lodr` uses both. See [decoders.md](../decoders.md) for what each mode does and how to set `lm_alpha` and `lm_beta`.

Add the models to the same config file used for training. `vocab_size` must equal the tokenizer's vocabulary size, because the LM returns scores indexed against the transducer's own tokens:

```yaml
lm_config:
  external_config:
    class_name: tensorflow_asr.models.lm.lstm_language_model>LSTMLanguageModel
    config:
      vocab_size: 1000
      embed_dim: 512
      units: 2048
      nlayers: 2
      tie_embeddings: True
  internal_config:
    class_name: tensorflow_asr.models.lm.bigram_language_model>BigramLanguageModel
    config:
      vocab_size: 1000
      blank: 0
```

### 7.1 Internal language model (for `lm_type: lodr`)

```bash
tensorflow_asr train_lm \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/datadir \
    --dataset-type=slice \
    --target=internal \
    --output=/path/to/modeldir/internal_lm.weights.h5
```

This one must be fitted on the ASR training transcripts, so there is no `--text-path` for it — the whole point is to approximate what the transducer already picked up from that exact text. `BigramLanguageModel` is fitted by counting in a single pass, which is its exact estimate, so `--epochs`, `--bs` and `--learning-rate` do nothing here.

### 7.2 External language model

Point `--text-path` at a corpus much larger than your transcripts. The published setups use the LibriSpeech LM corpus, ~800M words against the ~9M words of LibriSpeech transcripts:

```bash
wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz

tensorflow_asr train_lm \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/datadir \
    --dataset-type=slice \
    --target=external \
    --text-path=/path/to/librispeech-lm-norm.txt.gz \
    --max-lines=1000000 \
    --bs=128 \
    --epochs=1 \
    --output=/path/to/modeldir/lm.weights.h5
## See others params
tensorflow_asr train_lm --help
```

The `.gz` is read directly and streamed, so the several GB never has to be unpacked. `--max-lines` caps it for a quick first run; drop it to use the whole corpus. `--datadir` and `--dataset-type` are still required even though the text comes from `--text-path`, because the config is rendered with them.

Without `--text-path` it falls back to the training transcripts and warns, since that trains the external LM on the text the transducer already learned.

### 7.3 Use the weights

The weights are not named in the config. Pass them to `tensorflow_asr test`, which loads them into the models `lm_config` describes:

```bash
tensorflow_asr test \
    --config-path=/path/to/config.yml.j2 \
    --dataset-type=slice \
    --datadir=/path/to/datadir \
    --outputdir=/path/to/modeldir/tests \
    --h5=/path/to/modeldir/weights.h5 \
    --lm-h5=/path/to/modeldir/lm.weights.h5 \
    --internal-lm-h5=/path/to/modeldir/internal_lm.weights.h5
```