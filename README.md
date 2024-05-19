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
  - [Installing from source (recommended)](#installing-from-source-recommended)
  - [Installing via PyPi](#installing-via-pypi)
  - [Installing for development](#installing-for-development)
  - [Install for Apple Sillicon](#install-for-apple-sillicon)
  - [Running in a container](#running-in-a-container)
- [Training \& Testing Tutorial](#training--testing-tutorial)
- [Features Extraction](#features-extraction)
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

### Installing from source (recommended)

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
# Tensorflow 2.x (with 2.x.x >= 2.5.1)
pip3 install ".[tf2.x]" # or ".[tf2.x-gpu]"
```

For anaconda3:

```bash
conda create -y -n tfasr tensorflow-gpu python=3.8 # tensorflow if using CPU, this makes sure conda install all dependencies for tensorflow
conda activate tfasr
pip install -U tensorflow-gpu # upgrade to latest version of tensorflow
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
# Tensorflow 2.x (with 2.x.x >= 2.5.1)
pip3 install ".[tf2.x]" # or ".[tf2.x-gpu]"
```

### Installing via PyPi

```bash
# Tensorflow 2.x (with 2.x >= 2.3)
pip3 install "TensorFlowASR[tf2.x]" # or pip3 install "TensorFlowASR[tf2.x-gpu]"
```

### Installing for development

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
pip3 install -e ".[dev]"
pip3 install -e ".[tf2.x]" # or ".[tf2.x-gpu]" or ".[tf2.x-apple]" for apple m1 machine
```

### Install for Apple Sillicon

Due to tensorflow-text is not built for Apple Sillicon, we need to install it with the prebuilt wheel file from [sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon)

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
pip3 install -e "." # or pip3 install -e ".[dev] for development # or pip3 install "TensorFlowASR[dev]" from PyPi
pip3 install tensorflow~=2.14.0 # change minor version if you want
```

Do this after installing TensorFlowASR with tensorflow above

```bash
TF_VERSION="$(python3 -c 'import tensorflow; print(tensorflow.__version__)')" && \
TF_VERSION_MAJOR="$(echo $TF_VERSION | cut -d'.' -f1,2)" && \
PY_VERSION="$(python3 -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(f"{major}{minor}");')" && \
URL="https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon" && \
pip3 install "${URL}/releases/download/v${TF_VERSION_MAJOR}/tensorflow_text-${TF_VERSION_MAJOR}.0-cp${PY_VERSION}-cp${PY_VERSION}-macosx_11_0_arm64.whl"
```

### Running in a container

```bash
docker-compose up -d
```



## Training & Testing Tutorial

- For training, please read [tutorial_training](./docs/tutorials/training.md)
- For testing, please read [tutorial_testing](./docs/tutorials/testing.md)

**FYI**: Keras builtin training uses **infinite dataset**, which avoids the potential last partial batch.

See [examples](./examples/) for some predefined ASR models and results

## Features Extraction

See [features_extraction](./tensorflow_asr/featurizers/README.md)

## Augmentations

See [augmentations](./tensorflow_asr/augmentations/README.md)

## TFLite Convertion

After converting to tflite, the tflite model is like a function that transforms directly from an **audio signal** to **text and tokens**

See [tflite_convertion](./docs/tutorials/tflite.md)

## Pretrained Models

Go to [drive](https://drive.google.com/drive/folders/1BD0AK30n8hc-yR28C5FW3LqzZxtLOQfl?usp=sharing)

## Corpus Sources

### English

| **Name**     | **Source**                                                         | **Hours** |
| :----------- | :----------------------------------------------------------------- | :-------- |
| LibriSpeech  | [LibriSpeech](http://www.openslr.org/12)                           | 970h      |
| Common Voice | [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org) | 1932h     |

### Vietnamese

| **Name**                               | **Source**                                                                             | **Hours** |
| :------------------------------------- | :------------------------------------------------------------------------------------- | :-------- |
| Vivos                                  | [https://ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos)                   | 15h       |
| InfoRe Technology 1                    | [InfoRe1 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/25hours.zip)    | 25h       |
| InfoRe Technology 2 (used in VLSP2019) | [InfoRe2 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/audiobooks.zip) | 415h      |

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
