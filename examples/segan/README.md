# Speech Enhancement Generative Adversarial Network

References: [https://arxiv.org/abs/1703.09452](https://arxiv.org/abs/1703.09452)

## Model YAML Config Structure

```yaml
model_config:
  g_enc_depths: [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
  d_num_fmaps: [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
  kwidth:           31
  ratio:            2
```

## Training and Testing

See `python examples/segan/run_segan.py --help`