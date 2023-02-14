# Sentencepiece Contextnet Transducer

- [Sentencepiece Contextnet Transducer](#sentencepiece-contextnet-transducer)
  - [2023-02-14](#2023-02-14)


## 2023-02-14

Config:

```python
config = """
{% set repodir = "/path/to/TensorFlowASR" %}
{% set modeldir = "/path/to/models/sp1k-contextnet/20230214" %}
{% set datadir = "/path/to/librispeech/tfrecords" %}

model_config:
  name: contextnet
  encoder_alpha: 0.5
  encoder_blocks:
    # C0
    - nlayers: 1
      kernel_size: 5
      filters: 256
      strides: 1
      residual: False
      activation: silu
      padding: causal
    # C1-C2
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    # C3
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 2
      residual: True
      activation: silu
      padding: causal
    # C4-C6
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    # C7
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 2
      residual: True
      activation: silu
      padding: causal
    # C8 - C10
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 256
      strides: 1
      residual: True
      activation: silu
      padding: causal
    # C11 - C13
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    # C14
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 2
      residual: True
      activation: silu
      padding: causal
    # C15 - C21
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    - nlayers: 5
      kernel_size: 5
      filters: 512
      strides: 1
      residual: True
      activation: silu
      padding: causal
    # C22
    - nlayers: 1
      kernel_size: 5
      filters: 640
      strides: 1
      residual: False
      activation: silu
      padding: causal
  prediction_label_encode_mode: embedding
  prediction_embed_dim: 640
  prediction_num_rnns: 1
  prediction_rnn_units: 512
  prediction_rnn_type: lstm
  prediction_rnn_implementation: 2
  prediction_rnn_unroll: False
  prediction_layer_norm: True
  prediction_projection_units: 0
  joint_dim: 512
  prejoint_encoder_linear: True
  prejoint_prediction_linear: True
  postjoint_linear: False
  joint_activation: tanh
  joint_mode: add

speech_config:
  sample_rate: 16000
  frame_ms: 25
  stride_ms: 10
  num_feature_bins: 80
  feature_type: log_mel_spectrogram

decoder_config:
  type: sentencepiece

  blank_index: 0
  pad_token: "<pad>"
  pad_index: 0
  unknown_token: "<unk>"
  unknown_index: 1
  bos_token: "<s>"
  bos_index: 2
  eos_token: "</s>"
  eos_index: 3

  beam_width: 0
  norm_score: True
  lm_config: null

  model_type: bpe
  vocabulary: {{repodir}}/vocabularies/librispeech/sentencepiece/train_bpe_1000.model
  vocab_size: 1000
  max_token_length: 50
  max_unique_chars: 1000
  reserved_tokens: null
  normalization_form: NFKC
  num_iterations: 4

  corpus_files: null

learning_config:
  train_dataset_config:
    enabled: True
    use_tf: True
    augmentation_config:
      feature_augment:
        time_masking:
          prob: 1.0
          num_masks: 10
          mask_factor: 100
          p_upperbound: 0.05
          mask_value: mean
        freq_masking:
          prob: 1.0
          num_masks: 1
          mask_factor: 27
          mask_value: mean
    data_paths: null
    tfrecords_dir: {{datadir}}
    shuffle: True
    cache: False
    buffer_size: 1000
    drop_remainder: True
    stage: train
    metadata: {{repodir}}/vocabularies/librispeech/sentencepiece/train_bpe_1000.metadata.json

  eval_dataset_config:
    enabled: False
    use_tf: True
    data_paths: null
    tfrecords_dir: null
    shuffle: False
    cache: True
    buffer_size: 100
    drop_remainder: True
    stage: eval
    metadata: null

  test_dataset_config:
    enabled: False
    use_tf: True
    data_paths: null
    tfrecords_dir: null
    shuffle: False
    cache: True
    buffer_size: 100
    drop_remainder: True
    stage: test

  optimizer_config:
    beta_1: 0.9
    beta_2: 0.98
    epsilon: 1e-9

  learning_rate_config:
    warmup_steps: 15000
    max_lr: 0.0025

  apply_gwn_config:
    predict_net_step: 20000
    predict_net_stddev: 0.075
    joint_net_step: 20000
    joint_net_stddev: 0.075

  running_config:
    batch_size: 6
    num_epochs: 300
    checkpoint:
      filepath: {{modeldir}}/checkpoints/{epoch:02d}.h5
      save_best_only: False
      save_weights_only: True
      save_freq: epoch
      options:
        experimental_enable_async_checkpoint: True
    backup_and_restore:
      backup_dir: {{modeldir}}/states
    tensorboard:
      log_dir: {{modeldir}}/tensorboard
      write_graph: False
      write_images: False
      update_freq: epoch
      profile_batch: 100
"""
with open("/path/to/config.j2", "w") as f:
    f.write(config)
```

Training:

```bash
python /path/to/TensorFlowASR/examples/transducer/contextnet/train.py \
    --config-path=/path/to/config.j2 \
    --mxp=strict \
    --jit-compile \
    --tfrecords
```

Testing:

```bash
python /path/to/TensorFlowASR/examples/transducer/contextnet/test.py \
    --config-path=/path/to/config.j2 \
    --saved=/path/to/models/sp1k-contextnet/20230214/checkpoints/25.h5 \
    --output=/path/to/models/sp1k-contextnet/20230214/tests/25.tsv \
    --bs=1
```

RNNT Loss Curves:



Error Rates:

| Dataset                |  Mode  | Batch size | Epoch | WER (%) | CER (%) |
| :--------------------- | :----: | :--------: | :---: | :-----: | :-----: |
| librispeech-test-clean | greedy |     1      |  25   |         |         |
| librispeech-test-other | greedy |     1      |  25   |         |         |