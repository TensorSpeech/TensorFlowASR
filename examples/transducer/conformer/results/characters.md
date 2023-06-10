# Characters Conformer Transducer

- [Characters Conformer Transducer](#characters-conformer-transducer)
  - [2023-02-12](#2023-02-12)


## 2023-02-12

Config:

```python
config = """
{% set repodir = "/path/to/TensorFlowASR" %}
{% set modeldir = "/path/to/models/char-conformer/20230212" %}
{% set datadir = "/path/to/librispeech/tfrecords" %}

model_config:
  name: conformer
  encoder_subsampling:
    type: conv2d
    nlayers: 2
    filters: 144
    kernel_size: 3
    strides: 2
    padding: same
    norm: batch
    activation: swish
  encoder_dmodel: 144
  encoder_num_blocks: 16
  encoder_head_size: 36 # == dmodel // num_heads
  encoder_num_heads: 4
  encoder_mha_type: relmha
  encoder_use_attention_causal_mask: False
  encoder_kernel_size: 32
  encoder_fc_factor: 0.5
  encoder_dropout: 0.1
  encoder_padding: causal
  prediction_label_encode_mode: embedding
  prediction_embed_dim: 320
  prediction_num_rnns: 1
  prediction_rnn_units: 320
  prediction_rnn_type: lstm
  prediction_rnn_implementation: 2
  prediction_rnn_unroll: False
  prediction_layer_norm: False
  prediction_projection_units: 0
  joint_dim: 320
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
  normalize_feature: False

decoder_config:
  type: characters
  blank_index: 0
  beam_width: 0
  norm_score: True
  lm_config: null
  vocabulary: {{repodir}}/vocabularies/librispeech/characters/english.characters
  corpus_files: null

learning_config:
  train_dataset_config:
    enabled: True
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
    metadata: {{repodir}}/vocabularies/librispeech/characters/metadata.json

  eval_dataset_config:
    enabled: False
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
    warmup_steps: 10000
    max_lr_numerator: 0.05

  apply_gwn_config:
    predict_net_step: 20000
    predict_net_stddev: 0.075

  running_config:
    batch_size: 4
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
python /path/to/TensorFlowASR/examples/transducer/conformer/train.py \
    --config-path=/path/to/config.j2 \
    --mxp=strict \
    --jit-compile \
    --tfrecords
```

Testing:

```bash
python /path/to/TensorFlowASR/examples/transducer/conformer/test.py \
    --config-path=/path/to/config.j2 \
    --saved=/path/to/models/char-conformer/20230212/checkpoints/25.h5 \
    --output=/path/to/models/char-conformer/20230212/tests/25.tsv \
    --bs=1
```

RNNT Loss Curves:



Error Rates:

| Dataset                |  Mode  | Batch size | Epoch | WER (%) | CER (%) |
| :--------------------- | :----: | :--------: | :---: | :-----: | :-----: |
| librispeech-test-clean | greedy |     1      |  25   |         |         |
| librispeech-test-other | greedy |     1      |  25   |         |         |