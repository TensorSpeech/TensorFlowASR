# Conformer: Convolution-augmented Transformer for Speech Recognition

Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100)

![Conformer Architecture](./figs/arch.png)

## Example Model YAML Config

Go to [config.yml](./config.yml)

## Usage

Training, see `python examples/conformer/train_*.py --help`

Testing, see `python examples/conformer/test_*.py --help`

TFLite Conversion, see `python examples/conformer/tflite_*.py --help`

## Conformer Subwords - Results on LibriSpeech

**Summary**

- Number of subwords: 1031
- Maxium length of a subword: 4
- Subwords corpus: all training sets, dev sets and test-clean
- Number of parameters: 10,341,639
- Positional Encoding Type: sinusoid concatenation
- Trained on: 4 RTX 2080Ti 11G

**Pretrained and Config**, go to [drive](https://drive.google.com/drive/folders/1VAihgSB5vGXwIVTl3hkUk95joxY1YbfW?usp=sharing)

**Transducer Loss**

<img src="./figs/subword_conformer_loss.svg" alt="conformer_subword" width="300px" />

**Error Rates**

| **Test-clean** | Test batch size |  WER (%)   |  CER (%)   |
| :------------: | :-------------: | :--------: | :--------: |
|    _Greedy_    |        1        | 6.37933683 | 2.4757576  |
|  _Greedy V2_   |        1        | 7.86670732 | 2.82563138 |

| **Test-other** | Test batch size |  WER (%)   |  CER (%)   |
| :------------: | :-------------: | :--------: | :--------: |
|    _Greedy_    |        1        | 15.7308521 | 7.67273521 |
