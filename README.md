<h1 align="center">
TensorFlowASR :zap:
</h1>
<p align="center">
<a href="https://github.com/TensorSpeech/TensorFlowASR/blob/main/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/TensorSpeech/TensorFlowASR?logo=apache&logoColor=green">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.12.0-orange?logo=tensorflow">
<a href="https://pypi.org/project/TensorFlowASR/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/TensorFlowASR?color=%234285F4&label=release&logo=pypi&logoColor=%234285F4">
</a>
</p>
<h2 align="center">
Almost State-of-the-art Automatic Speech Recognition in Tensorflow 2
</h2>

<p align="center">
TensorFlowASR implements some automatic speech recognition architectures such as DeepSpeech2, Jasper, RNN Transducer, ContextNet, Conformer, etc. These models can be converted to TFLite to reduce memory and computation for deployment :smile:
</p>

## What's New?

## Table of Contents

<!-- TOC -->

- [What's New?](#whats-new)
- [Table of Contents](#table-of-contents)
- [:yum: Supported Models](#yum-supported-models)
  - [Baselines](#baselines)
  - [Publications](#publications)
- [Installation](#installation)
- [Training \& Testing Tutorial](#training--testing-tutorial)
- [Features Extraction](#features-extraction)
- [Decoders](#decoders)
- [Augmentations](#augmentations)
- [TFLite Convertion](#tflite-convertion)
- [Pretrained Models](#pretrained-models)
- [Corpus Sources](#corpus-sources)
  - [English](#english)
  - [Vietnamese](#vietnamese)
- [How to contribute](#how-to-contribute)
- [References \& Credits](#references--credits)
- [Contact](#contact)

<!-- /TOC -->

## :yum: Supported Models

### Baselines

- **Transducer Models** (End2end models using RNNT Loss for training, currently supported Conformer, ContextNet, Streaming Transducer)
- **CTCModel** (End2end models using CTC Loss for training, currently supported DeepSpeech2, Jasper)

### Publications

- **Conformer Transducer** (Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100))
  See [examples/models/transducer/conformer](./examples/models/transducer/conformer)
- **Streaming Conformer** (Reference: [http://arxiv.org/abs/2010.11395](http://arxiv.org/abs/2010.11395))
  See [examples/models/transducer/conformer](./examples/models/transducer/conformer)
- **ContextNet** (Reference: [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191))
  See [examples/models/transducer/contextnet](./examples/models/transducer/contextnet)
- **RNN Transducer** (Reference: [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621))
  See [examples/models/transducer/rnnt](./examples/models/transducer/rnnt)
- **Deep Speech 2** (Reference: [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595))
  See [examples/models/ctc/deepspeech2](./examples/models/ctc/deepspeech2)
- **Jasper** (Reference: [https://arxiv.org/abs/1904.03288](https://arxiv.org/abs/1904.03288))
  See [examples/models/ctc/jasper](./examples/models/ctc/jasper)

## Installation

For training and testing, you should use `git clone` for installing necessary packages from other authors (`ctc_decoders`, `rnnt_loss`, etc.)

TensorFlowASR uses [uv](https://docs.astral.sh/uv/) as its package manager and requires python 3.12 or 3.13.

A plain `uv sync` installs `tensorflow` + `tensorflow-text`, which is all that CPU
and Apple Silicon need. Accelerators are opt-in extras:

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
uv sync                 # CPU / Apple Silicon
uv sync --extra cuda    # NVIDIA GPU
uv sync --extra dev     # add the development tooling
```

Extras are declared in `pyproject.toml`:

| Extra  | Contents |
| ------ | -------- |
| _(none)_ | `tensorflow` + `tensorflow-text`, works on CPU and Apple Silicon |
| `cuda` | `tensorflow[and-cuda]` for NVIDIA GPUs |
| `dev`  | `pytest`, `ruff`, `pre-commit`, plotting and export tooling |

Run commands inside the environment with `uv run`, e.g. `uv run pytest` or
`uv run tensorflow_asr --help`.

**Cloud TPU** is not an extra:

```bash
uv sync && ./scripts/install_tpu.sh
```

`tensorflow-tpu` ships its own `tensorflow` distribution, and `tensorflow` is a base
dependency, so an extra adding it would install both and leave whichever landed last in
place. The script uninstalls the stock build first, then installs the TPU one. Re-run it
after any later `uv sync`, which puts the stock `tensorflow` back.

> **TPU note:** `tensorflow-tpu` ships its own `tensorflow` distribution and
> overwrites the base one. This matches the previous `setup.sh tpu` behaviour,
> which uninstalled `tensorflow` and force-installed `tensorflow-tpu` over it.

**Running in a container**

```bash
docker-compose up -d
```


## Training & Testing Tutorial

- For training, please read [tutorial_training](./docs/tutorials/training.md)
- For testing, please read [tutorial_testing](./docs/tutorials/testing.md)

**FYI**: Keras builtin training uses **infinite dataset**, which avoids the potential last partial batch.

See [examples](./examples/) for some predefined ASR models and results

## Features Extraction

See [features_extraction](./tensorflow_asr/features/README.md)

## Decoders

Greedy and beam search decoding for CTC and Transducer models, including the ALSD++ transducer beam search

See [decoders](./docs/decoders.md)

## Augmentations

See [augmentations](./tensorflow_asr/augmentations/README.md)

## TFLite Convertion

After converting to tflite, the tflite model is like a function that transforms directly from an **audio signal** to **text and tokens**

See [tflite_convertion](./docs/tutorials/tflite.md)

## Pretrained Models

See the results on each example folder, e.g. [./examples/models//transducer/conformer/results/sentencepiece/README.md](./examples/models//transducer/conformer/results/sentencepiece/README.md)

## Corpus Sources

### English

| **Name**     | **Source**                                                         | **Hours** |
| :----------- | :----------------------------------------------------------------- | :-------- |
| LibriSpeech  | [LibriSpeech](http://www.openslr.org/12)                           | 970h      |
| Common Voice | [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org) | 1932h     |

### Vietnamese

| **Name**                               | **Source**                                                                                                           | **Hours** |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :-------- |
| Vivos                                  | [https://ailab.hcmus.edu.vn/vivos](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr) | 15h       |
| InfoRe Technology 1                    | [InfoRe1 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/datasets/infore/25hours.zip)                  | 25h       |
| InfoRe Technology 2 (used in VLSP2019) | [InfoRe2 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/datasets/infore/audiobooks.zip)               | 415h      |
| VietBud500                             | [https://huggingface.co/datasets/linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500)     | 500h      |

## How to contribute

1. Fork the project
2. [Install for development](#installing-for-development)
3. Create a branch
4. Make a pull request to this repo

## References & Credits

1. [NVIDIA OpenSeq2Seq Toolkit](https://github.com/NVIDIA/OpenSeq2Seq)
2. [https://github.com/noahchalifour/warp-transducer](https://github.com/noahchalifour/warp-transducer)
3. [Sequence Transduction with Recurrent Neural Network](https://arxiv.org/abs/1211.3711)
4. [End-to-End Speech Processing Toolkit in PyTorch](https://github.com/espnet/espnet)
5. [https://github.com/iankur/ContextNet](https://github.com/iankur/ContextNet)

## Contact

Huy Le Nguyen

Email: nlhuy.cs.16@gmail.com
