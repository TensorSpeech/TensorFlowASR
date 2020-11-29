# Streaming End-to-end Speech Recognition For Mobile Devices

Reference: [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621)

## Example Model YAML Config

```yaml
speech_config:
  sample_rate: 16000
  frame_ms: 25
  stride_ms: 10
  feature_type: log_mel_spectrogram
  num_feature_bins: 80
  preemphasis: 0.97
  normalize_signal: True
  normalize_feature: True
  normalize_per_feature: False

decoder_config:
  vocabulary: null
  target_vocab_size: 1024
  max_subword_length: 4
  blank_at_zero: True
  beam_width: 5
  norm_score: True

model_config:
  name: streaming_transducer
  reduction_factor: 2
  reduction_positions: [1]
  encoder_dim: 320
  encoder_units: 1024
  encoder_layers: 7
  encoder_layer_norm: True
  encoder_type: lstm
  embed_dim: 320
  embed_dropout: 0.1
  num_rnns: 1
  rnn_units: 320
  rnn_type: lstm
  layer_norm: True
  joint_dim: 320

learning_config:
  augmentations:
    after:
      time_masking:
        num_masks: 10
        mask_factor: 100
        p_upperbound: 0.2
      freq_masking:
        num_masks: 1
        mask_factor: 27

  dataset_config:
    train_paths: ...
    eval_paths: ...
    test_paths: ...
    tfrecords_dir: ...

  running_config:
    batch_size: 4
    num_epochs: 22
    outdir: ...
    log_interval_steps: 400
    save_interval_steps: 400
    eval_interval_steps: 1000
```

## Usage

Training, see `python examples/streamingTransducer/train_streaming_transducer.py --help`

Testing, see `python examples/streamingTransducer/train_streaming_transducer.py --help`

TFLite Conversion, see `python examples/streamingTransducer/tflite_streaming_transducer.py --help`
