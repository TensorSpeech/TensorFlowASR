**Table of Contents**
- [LibriSpeech](#librispeech)
  - [I. Small + SentencePiece 256](#i-small--sentencepiece-256)
    - [Training](#training)
      - [1. Epoch Loss](#1-epoch-loss)
      - [2. Batch Loss](#2-batch-loss)
      - [3. Learning Rate](#3-learning-rate)
    - [Pretrained Model](#pretrained-model)
    - [Results](#results)
- [VietBud500](#vietbud500)
  - [I. Small + SentencePiece 256](#i-small--sentencepiece-256-1)
    - [Training](#training-1)
      - [1. Epoch Loss](#1-epoch-loss-1)
      - [2. Batch Loss](#2-batch-loss-1)
      - [3. Learning Rate](#3-learning-rate-1)
    - [Pretrained Model](#pretrained-model-1)
    - [Results](#results-1)


# LibriSpeech

## I. Small + SentencePiece 256

| Category          | Description                                                |
| :---------------- | :--------------------------------------------------------- |
| Config            | [small.yml.j2](../../small.yml.j2)                         |
| Tensorflow        | **2.18.0**                                                 |
| Device            | Google Cloud TPUs v4-8                                     |
| Mixed Precision   | strict                                                     |
| Global Batch Size | 8 * 4 * 8 = 256 (as 4 TPUs, 8 Gradient Accumulation Steps) |
| Max Epochs        | 450                                                        |


### Training

#### 1. Epoch Loss

![Epoch Loss](./figs/)

#### 2. Batch Loss

![Batch Loss](./figs/)

#### 3. Learning Rate

![Learning Rate](./figs/)

### Pretrained Model

[Link]()

### Results


```json
[
  {
    "epoch": 157,
    "test-clean": {
    },
    "test-other": {
    }
  }
]
```

# VietBud500

## I. Small + SentencePiece 256

| Category          | Description                                                |
| :---------------- | :--------------------------------------------------------- |
| Config            | [small.yml.j2](../../small.yml.j2)                         |
| Tensorflow        | **2.18.0**                                                 |
| Device            | Google Cloud TPUs v4-8                                     |
| Mixed Precision   | strict                                                     |
| Global Batch Size | 8 * 4 * 8 = 256 (as 4 TPUs, 8 Gradient Accumulation Steps) |
| Max Epochs        | 450                                                        |

### Training

#### 1. Epoch Loss

![Epoch Loss](./figs/)

#### 2. Batch Loss

![Batch Loss](./figs/)

#### 3. Learning Rate

![Learning Rate](./figs/)

### Pretrained Model

[Link]()

### Results

```json
[

]
```