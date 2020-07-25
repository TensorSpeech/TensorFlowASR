# Conformer: Convolution-augmented Transformer for Speech Recognition

Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100)

![Conformer Architecture](./figs/arch.png)

## Example Model YAML Config

```yaml
model_config:
  name: conformer
  ffm_dim: 1024
  subsampling_kernel_size: 32
  subsampling_filters: 144
  subsampling_strides: 2
  subsampling_dropout: 0.1
  num_blocks: 16
  head_size: 36
  num_heads: 4
  kernel_size: 32
  fc_factor: 0.5
  dropout: 0.1
  embed_dim: 256
  embed_dropout: 0.0
  num_lstms: 1
  lstm_units: 320
  joint_dim: 1024
```
## Usage

See `python examples/conformer/run_conformer.py --help`
