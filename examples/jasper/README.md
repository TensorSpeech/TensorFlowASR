# Jasper

References: [https://arxiv.org/abs/1904.03288](https://arxiv.org/abs/1904.03288)

## Model YAML Config Structure

```yaml
model_config:
  name: jasper
  dense: True
  first_additional_block_channels: 256
  first_additional_block_kernels: 11
  first_additional_block_strides: 2
  first_additional_block_dilation: 1
  first_additional_block_dropout: 0.2
  nsubblocks: 3
  block_channels: [256, 384, 512, 640, 768]
  block_kernels: [11, 13, 17, 21, 25]
  block_dropout: [0.2, 0.2, 0.2, 0.3, 0.3]
  second_additional_block_channels: 896
  second_additional_block_kernels: 1
  second_additional_block_strides: 1
  second_additional_block_dilation: 2
  second_additional_block_dropout: 0.4
  third_additional_block_channels: 1024
  third_additional_block_kernels: 1
  third_additional_block_strides: 1
  third_additional_block_dilation: 1
  third_additional_block_dropout: 0.4
```

## Architecture

<img src="./figs/jasper_arch.png" alt="jasper_arch" width="800px" />

## Training and Testing

See `python examples/jasper/train_*.py --help`

See `python examples/jasper/test_*.py --help`

