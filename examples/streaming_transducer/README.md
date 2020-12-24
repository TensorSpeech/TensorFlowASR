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
  encoder_reductions:
    0: 3
    1: 2
  encoder_dmodel: 320
  encoder_rnn_type: lstm
  encoder_rnn_units: 1024
  encoder_nlayers: 8
  encoder_layer_norm: True
  prediction_embed_dim: 320
  prediction_embed_dropout: 0.0
  prediction_num_rnns: 2
  prediction_rnn_units: 1024
  prediction_rnn_type: lstm
  prediction_projection_units: 320
  prediction_layer_norm: True
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

Training, see `python examples/streamingTransducer/train_*.py --help`

Testing, see `python examples/streamingTransducer/test_*.py --help`

TFLite Conversion, see `python examples/streamingTransducer/tflite_*.py --help`
