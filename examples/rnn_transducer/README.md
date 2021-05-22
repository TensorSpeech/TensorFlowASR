# Streaming End-to-end Speech Recognition For Mobile Devices

Reference: [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621)

## Example Model YAML Config

Go to [config.yml](./config.yml)

## Usage

Training, see `python examples/streamingTransducer/train_*.py --help`

Testing, see `python examples/streamingTransducer/test_*.py --help`

TFLite Conversion, see `python examples/streamingTransducer/tflite_*.py --help`

## RNN Transducer Subwords - Results on LibriSpeech

**Summary**

- Number of subwords: 1008
- Maximum length of a subword: 10
- Subwords corpus: all training sets
- Number of parameters: 54,914,480
- Number of epochs: 21
- Train on: 8 Google Colab TPUs
- Train hours: 10.5 days uncontinuous (each day I trained 2 epoch because colab only allows 12 hours/day and 1 epoch required 4.5 hours) => 94.5 hours continuous (3.9375 days)

**Pretrained and Config**, go to [drive](https://drive.google.com/drive/folders/1rYpiYF0F9JIsAKN2DCFFtEdfNzVbBLHe?usp=sharing)

**Epoch Transducer Loss**

<img src="./figs/subword_rnnt_loss.svg" alt="subword_rnnt_loss" width="300px" />

**Epoch Learning Rate**

<img src="./figs/epoch_learning_rate.svg" alt="epoch_learning_rate" width="300px" />

**Error Rates**

| **Test-clean** | Test batch size | Epoch |      WER (%)      |      CER (%)      |
| :------------: | :-------------: | :---: | :---------------: | :---------------: |
|    _Greedy_    |        8        |  21   | 13.13907504081726 | 6.023869663476944 |
|    _Greedy_    |        8        |  25   | 12.79481202363968 | 5.671864375472069 |