# Sentencepiece DeepSpeech2


- [Sentencepiece DeepSpeech2](#sentencepiece-deepspeech2)
  - [2023-02-12](#2023-02-12)



## 2023-02-12

Config:

```python
config = """
{% set repodir = "/path/to/TensorFlowASR" %}
{% set modeldir = "/path/to/models/sp1k-deepspeech2/20230212" %}
{% set datadir = "/path/to/librispeech/tfrecords" %}

model_config:
  name: deepspeech2
  conv_type: conv2d
  conv_kernels: [[11, 41], [11, 21], [11, 11]]
  conv_strides: [[3, 2], [2, 2], [1, 2]]
  conv_filters: [32, 32, 96]
  conv_padding: same
  conv_dropout: 0.1
  rnn_nlayers: 7
  rnn_type: lstm
  rnn_bn_type: bn
  rnn_units: 512
  rnn_bidirectional: True
  rnn_unroll: False
  rnn_rowconv: 0
  rnn_dropout: 0.1
  fc_nlayers: 0
  fc_units: 1024
  fc_dropout: 0.1

speech_config:
  sample_rate: 16000
  frame_ms: 25
  stride_ms: 10
  num_feature_bins: 128
  feature_type: spectrogram

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

learning_config:
  train_dataset_config:
    enabled: True
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
    class_name: adam
    config:
      beta_1: 0.9
      beta_2: 0.98
      epsilon: 1e-9

  running_config:
    batch_size: 8
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
python /path/to/TensorFlowASR/examples/ctc/deepspeech2/train.py \
    --config-path=/path/to/config.j2 \
    --mxp=strict \
    --jit-compile \
    --tfrecords
```

Testing:

```bash
python /path/to/TensorFlowASR/examples/ctc/deepspeech2/test.py \
    --config-path=/path/to/config.j2 \
    --saved=/path/to/models/sp1k-deepspeech2/20230212/checkpoints/25.h5 \
    --output=/path/to/models/sp1k-deepspeech2/20230212/tests/25.tsv \
    --bs=1
```

RNNT Loss Curves:



Error Rates:

| Dataset                | Mode                     | Batch size | Epoch | WER (%) | CER (%) |
| :--------------------- | :----------------------- | :--------: | :---: | :-----: | :-----: |
| librispeech-test-clean | greedy                   |     1      |  25   |         |         |
| librispeech-test-clean | beamsearch with size 500 |     1      |  25   |         |         |
| librispeech-test-other | greedy                   |     1      |  25   |         |         |
| librispeech-test-other | beamsearch with size 500 |     1      |  25   |         |         |