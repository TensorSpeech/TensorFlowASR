# Conformer: Convolution-augmented Transformer for Speech Recognition

Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100)

![Conformer Architecture](./figs/arch.png)

## Example Model YAML Config

Go to [config.yml](./config.yml)

## Usage

Training, see `python examples/conformer/train.py --help`

Testing, see `python examples/conformer/test.py --help`

TFLite Conversion, see `python examples/conformer/inference/gen_tflite_model.py --help`

## WordPiece Conformer - Results on LibriSpeech

| **Name**                 | **Description**              |
| :----------------------- | :--------------------------- |
| Number of tokens         | 1000                         |
| Maxium length of a token | 50                           |
| WordPiece Corpus         | All training transcripts.tsv |
| Trained on               | 8 Google Colab TPUs          |
| Training hours           |                              |

**Pretrained and Config**, go to [drive](https://drive.google.com/drive/folders/1VAihgSB5vGXwIVTl3hkUk95joxY1YbfW?usp=sharing)

**Epoch RNNT Loss**

<img src="./figs/subword_conformer_loss.svg" alt="conformer_subword" width="300px" />

**Error Rates**

| **Test-clean** | Test batch size | Epoch | WER (%) | CER (%) |
| :------------: | :-------------: | :---: | :-----: | :-----: |
|    _Greedy_    |        1        |  50   |         |         |

| **Test-other** | Test batch size | Epoch | WER (%) | CER (%) |
| :------------: | :-------------: | :---: | :-----: | :-----: |
|    _Greedy_    |        1        |  50   |         |         |
