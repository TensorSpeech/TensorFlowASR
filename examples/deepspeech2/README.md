# Deep Speech 2

References: [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595)

## Model YAML Config Structure

```yaml
model_config:
  conv_conf:
    conv_type: 2
    conv_kernels: [[11, 41], [11, 21], [11, 11]]
    conv_strides: [[2, 2], [1, 2], [1, 2]]
    conv_filters: [32, 32, 96]
    conv_dropout: 0
  rnn_conf:
    rnn_layers:        5
    rnn_type:          lstm
    rnn_units:         512
    rnn_bidirectional: True
    rnn_rowconv:       False
    rnn_dropout:       0
  fc_conf:
    fc_units: [1024]
    fc_dropout: 0
```

## Architecture

<img src="./figs/ds2_arch.png" alt="ds2_arch" width="500px" />

## Training and Testing

See `python examples/deepspeech2/run_ds2.py --help`

## Results on VIVOS Dataset

* Features: Spectrogram with `80` frequency channels
* KenLM: `alpha = 2.0` and `beta = 1.0`
* Epochs: `20`
* Train set split ratio: `90:10`
* Augmentation: `None`
* Model architecture: same as [vivos.yaml](./configs/vivos.yml)

**CTC Loss**

<img src="./figs/ds2_vivos_ctc_loss.svg" alt="ds2_vivos_ctc_loss" width="300px" />

**Error rates**

|                 |    WER (%)     |    CER (%)     |
| :-------------- | :------------: | :------------: |
| *BeamSearch*    |    43.75243    |   17.991581    |
| *BeamSearch LM* | **20.7561836** | **11.0304441** |