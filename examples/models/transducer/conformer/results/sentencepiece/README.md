**Table of Contents**
- [SentencePiece 1k + Small + LibriSpeech](#sentencepiece-1k--small--librispeech)
    - [Training Loss](#training-loss)
      - [1. Epoch Loss](#1-epoch-loss)
      - [2. Batch Loss](#2-batch-loss)
    - [Training Learning Rate](#training-learning-rate)
    - [Results](#results)


# SentencePiece 1k + Small + LibriSpeech


| Category      | Description                        |
| :------------ | :--------------------------------- |
| Config        | [small.yml.j2](../../small.yml.j2) |
| Tensorflow    | **2.13.x**                         |
| Device        | Google Colab TPUs                  |
| Training time |                                    |


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
  },
]
```
