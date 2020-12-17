<h1 align="center">
<p>TensorFlowASR :zap:</p>
<p align="center">
<a href="https://github.com/TensorSpeech/TensorFlowASR/blob/main/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/TensorSpeech/TensorFlowASR?logo=apache&logoColor=green">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.3.0-orange?logo=tensorflow">
<a href="https://pypi.org/project/TensorFlowASR/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/TensorFlowASR?color=%234285F4&label=release&logo=pypi&logoColor=%234285F4">
</a>
</p>
</h1>
<h2 align="center">
<p>Almost State-of-the-art Automatic Speech Recognition in Tensorflow 2</p>
</h2>

<p align="center">
TensorFlowASR implements some automatic speech recognition architectures such as DeepSpeech2, Jasper, ContextNet, Conformer, etc. These models can be converted to TFLite to reduce memory and computation for deployment :smile:
</p>

## What's New?

- (12/17/2020) Supported ContextNet [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191)
- (12/12/2020) Add support for using masking
- (11/14/2020) Supported Gradient Accumulation for Training in Larger Batch Size
- (11/3/2020) Reduce differences between `librosa.stft` and `tf.signal.stft`
- (10/31/2020) Update DeepSpeech2 and Supported Jasper [https://arxiv.org/abs/1904.03288](https://arxiv.org/abs/1904.03288)
- (10/18/2020) Supported Streaming Transducer [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621)

## Table of Contents

<!-- TOC -->

- [What's New?](#whats-new)
- [Table of Contents](#table-of-contents)
- [:yum: Supported Models](#yum-supported-models)
  - [Baselines](#baselines)
  - [Publications](#publications)
- [Installation](#installation)
  - [Installing via PyPi](#installing-via-pypi)
  - [Installing from source](#installing-from-source)
- [Setup training and testing](#setup-training-and-testing)
- [TFLite Convertion](#tflite-convertion)
- [Features Extraction](#features-extraction)
- [Augmentations](#augmentations)
- [Training & Testing](#training--testing)
- [Corpus Sources and Pretrained Models](#corpus-sources-and-pretrained-models)
  - [English](#english)
  - [Vietnamese](#vietnamese)
  - [German](#german)
- [References & Credits](#references--credits)
- [Contact](#contact)

<!-- /TOC -->

## :yum: Supported Models

### Baselines

- **CTCModel** (End2end models using CTC Loss for training)
- **Transducer Models** (End2end models using RNNT Loss for training)

### Publications

- **Deep Speech 2** (Reference: [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595))
  See [examples/deepspeech2](./examples/deepspeech2)
- **Jasper** (Reference: [https://arxiv.org/abs/1904.03288](https://arxiv.org/abs/1904.03288))
  See [examples/jasper](./examples/jasper)
- **Conformer Transducer** (Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100))
  See [examples/conformer](./examples/conformer)
- **Streaming Transducer** (Reference: [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621))
  See [examples/streaming_transducer](./examples/streaming_transducer)
- **ContextNet** (Reference: [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191))
  See [examples/contextnet](./examples/contextnet)

## Installation

Install `tensorflow>=2.3.0` or `tf-nightly`.

For training and testing, you should use `git clone` for installing necessary packages from other authors (`ctc_decoders`, `rnnt_loss`, etc.)

### Installing via PyPi

Run `pip3 install -U TensorFlowASR`

### Installing from source

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
python3 setup.py install
```

For anaconda3:

```bash
conda create -y -n tfasr tensorflow-gpu python=3.7 # tensorflow if using CPU
conda activate tfasr
pip install -U tensorflow-gpu # upgrade to latest version of tensorflow
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
python setup.py install
```

## Setup training and testing

- For datasets, see [datasets](./tensorflow_asr/datasets/README.md)

- For _training, testing and using_ **CTC Models**, run `./scripts/install_ctc_decoders.sh`

- For _training_ **Transducer Models**, run `export CUDA_HOME=/usr/local/cuda && ./scripts/install_rnnt_loss.sh` (**Note**: only `export CUDA_HOME` when you have CUDA)

- For _mixed precision training_, use flag `--mxp` when running python scripts from [examples](./examples)

- For _enabling XLA_, run `TF_XLA_FLAGS=--tf_xla_auto_jit=2 python3 $path_to_py_script`)

- For _hiding warnings_, run `export TF_CPP_MIN_LOG_LEVEL=2` before running any examples

## TFLite Convertion

After converting to tflite, the tflite model is like a function that transforms directly from an **audio signal** to **unicode code points**, then we can convert unicode points to string.

1. Install `tf-nightly` using `pip install tf-nightly`
2. Build a model with the same architecture as the trained model _(if model has tflite argument, you must set it to True)_, then load the weights from trained model to the built model
3. Load `TFSpeechFeaturizer` and `TextFeaturizer` to model using function `add_featurizers`
4. Convert model's function to tflite as follows:

```python
func = model.make_tflite_function(greedy=True) # or False
concrete_func = func.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
```

5. Save the converted tflite model as follows:

```python
if not os.path.exists(os.path.dirname(tflite_path)):
    os.makedirs(os.path.dirname(tflite_path))
with open(tflite_path, "wb") as tflite_out:
    tflite_out.write(tflite_model)
```

5. Then the `.tflite` model is ready to be deployed

## Features Extraction

See [features_extraction](./tensorflow_asr/featurizers/README.md)

## Augmentations

See [augmentations](./tensorflow_asr/augmentations/README.md)

## Training & Testing

**Example YAML Config Structure**

```yaml
speech_config: ...
model_config: ...
decoder_config: ...
learning_config:
  augmentations: ...
  dataset_config:
    train_paths: ...
    eval_paths: ...
    test_paths: ...
    tfrecords_dir: ...
  optimizer_config: ...
  running_config:
    batch_size: 8
    num_epochs: 20
    outdir: ...
    log_interval_steps: 500
```

See [examples](./examples/) for some predefined ASR models and results

## Corpus Sources and Pretrained Models

For pretrained models, go to [drive](https://drive.google.com/drive/folders/1BD0AK30n8hc-yR28C5FW3LqzZxtLOQfl?usp=sharing)

### English

|   **Name**   |                             **Source**                             | **Hours** |
| :----------: | :----------------------------------------------------------------: | :-------: |
| LibriSpeech  |              [LibriSpeech](http://www.openslr.org/12)              |   970h    |
| Common Voice | [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org) |   1932h   |

### Vietnamese

|                **Name**                |                                       **Source**                                       | **Hours** |
| :------------------------------------: | :------------------------------------------------------------------------------------: | :-------: |
|                 Vivos                  |          [https://ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos)          |    15h    |
|          InfoRe Technology 1           |  [InfoRe1 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/25hours.zip)   |    25h    |
| InfoRe Technology 2 (used in VLSP2019) | [InfoRe2 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/audiobooks.zip) |   415h    |

### German

|   **Name**   |                             **Source**                              | **Hours** |
| :----------: | :-----------------------------------------------------------------: | :-------: |
| Common Voice | [https://commonvoice.mozilla.org/](https://commonvoice.mozilla.org) |   750h    |

## References & Credits

1. [NVIDIA OpenSeq2Seq Toolkit](https://github.com/NVIDIA/OpenSeq2Seq)
2. [https://github.com/noahchalifour/warp-transducer](https://github.com/noahchalifour/warp-transducer)
3. [Sequence Transduction with Recurrent Neural Network](https://arxiv.org/abs/1211.3711)
4. [End-to-End Speech Processing Toolkit in PyTorch](https://github.com/espnet/espnet)
5. [https://github.com/iankur/ContextNet](https://github.com/iankur/ContextNet)

## Contact

Huy Le Nguyen

Email: nlhuy.cs.16@gmail.com