**Table of Contents**
- [SentencePiece 1k + Small + LibriSpeech](#sentencepiece-1k--small--librispeech)
    - [Training Loss](#training-loss)
      - [1. Epoch Loss](#1-epoch-loss)
      - [2. Batch Loss](#2-batch-loss)
    - [Training Learning Rate](#training-learning-rate)
    - [Results](#results)


# SentencePiece 1k + Small + LibriSpeech


| Category          | Description                        |
| :---------------- | :--------------------------------- |
| Config            | [small.yml.j2](../../small.yml.j2) |
| Tensorflow        | **2.13.x**                         |
| Device            | Google Colab TPUs                  |
| Global Batch Size | 2 * 16 * 8 = 256 (as 8 TPUs)       |
| Max Epochs        | 300                                |
| Training time     |                                    |


### Training Loss

#### 1. Epoch Loss

![Epoch Loss](./figs/conformer-small-sp1k-epoch-loss.svg)

#### 2. Batch Loss

![Batch Loss](./figs/conformer-small-sp1k-batch-loss.svg)

### Training Learning Rate

![Learning Rate](./figs/conformer-small-sp1k-lr.svg)


### Results

Pretrain Model here: [link]()

```json
[
  {
    "epoch": 67,
    "test-clean": {
      "greedy": {
        "wer": 0.06362218502738892,
        "cer": 0.024043618797286257,
        "mer": 0.06317399762035165,
        "wil": 0.11067193097595462,
        "wip": 0.8893280690240454
      }
    },
    "test-other": {
      "greedy": {
        "wer": 0.15365951512141068,
        "cer": 0.07302077299290946,
        "mer": 0.1511444356748224,
        "wil": 0.25542074621900557,
        "wip": 0.7445792537809944
      }
    }
  },
]
```
