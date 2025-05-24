**Table of Contents**
- [\[English\] LibriSpeech](#english-librispeech)
  - [I. Small + SentencePiece 1k](#i-small--sentencepiece-1k)
    - [Config](#config)
    - [Results](#results)
- [\[Vietnamese\] VietBud500](#vietnamese-vietbud500)
  - [II. Small + Streaming + SentencePiece 1k](#ii-small--streaming--sentencepiece-1k)
    - [Config](#config-1)
    - [Results](#results-1)

<!-- ----------------------------------------------------- EN ------------------------------------------------------ -->

# [English] LibriSpeech

## I. Small + SentencePiece 1k

| Category          | Description                                                |
| :---------------- | :--------------------------------------------------------- |
| Config            | [small.yml.j2](../../small.yml.j2)                         |
| Tensorflow        | **2.18.0**                                                 |
| Device            | Google Cloud TPUs v4-8                                     |
| Mixed Precision   | strict                                                     |
| Global Batch Size | 4 * 4 * 8 = 128 (as 4 TPUs, 8 Gradient Accumulation Steps) |
| Max Epochs        | 300                                                        |

### Config

```jinja2
{% import "examples/datasets/librispeech/sentencepiece/sp.yml.j2" as decoder_config with context %}
{{decoder_config}}
{% import "examples/models/transducer/conformer/small.yml.j2" as config with context %}
{{config}}
```

### Results

| Epoch | Dataset    | decoding | wer      | cer      | mer      | wil      | wip      |
| :---- | :--------- | :------- | :------- | :------- | :------- | :------- | :------- |
| 157   | test-clean | greedy   | 0.062918 | 0.025361 | 0.062527 | 0.109992 | 0.890007 |
| 157   | test-other | greedy   | 0.142616 | 0.066839 | 0.140610 | 0.239201 | 0.760798 |

<!-- ----------------------------------------------------- VN ------------------------------------------------------ -->

# [Vietnamese] VietBud500

## II. Small + Streaming + SentencePiece 1k

| Category          | Description                                                |
| :---------------- | :--------------------------------------------------------- |
| Config            | [small-streaming.yml.j2](../../small-streaming.yml.j2)     |
| Tensorflow        | **2.18.0**                                                 |
| Device            | Google Cloud TPUs v4-8                                     |
| Mixed Precision   | strict                                                     |
| Global Batch Size | 8 * 4 * 8 = 256 (as 4 TPUs, 8 Gradient Accumulation Steps) |
| Max Epochs        | 300                                                        |

### Config

```jinja2
{% import "examples/datasets/vietbud500/sentencepiece/sp.yml.j2" as decoder_config with context %}
{{decoder_config}}
{% import "examples/models/transducer/conformer/small-streaming.yml.j2" as config with context %}
{{config}}
```

### Results

| Training      | Image                                                           |
| :------------ | :-------------------------------------------------------------- |
| Epoch Loss    | ![Epoch Loss](./figs/vietbud500-small-streaming-epoch-loss.svg) |
| Batch Loss    | ![Batch Loss](./figs/vietbud500-small-streaming-batch-loss.svg) |
| Learning Rate | ![Learning Rate](./figs/vietbud500-small-streaming-lr.svg)      |

| Epoch | decoding | wer      | cer      | mer     | wil      | wip      |
| :---- | :------- | :------- | :------- | :------ | :------- | :------- |
| 52    | greedy   | 0.053723 | 0.034548 | 0.05362 | 0.086421 | 0.913579 |

**Pretrained Model**: [Link](https://www.kaggle.com/models/lordh9072/tfasr-vietbud500-conformer-transducer/tensorFlow2/small-streaming)