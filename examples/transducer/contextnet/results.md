# Wordpiece Contextnet Transducer

- [Wordpiece Contextnet Transducer](#wordpiece-contextnet-transducer)
  - [LibriSpeech Only Data](#librispeech-only-data)
      - [Config](#config)
      - [Training](#training)
      - [Testing](#testing)


## LibriSpeech Only Data

#### Config

```python
config = """
{% set repodir = "/path/to/TensorFlowASR" %}
{% set modeldir = "/path/to/models/wp1k-contextnet/only-data" %}
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
  type: wordpiece

  blank_index: 0
  unknown_token: "<unk>"
  unknown_index: 1

  beam_width: 0
  norm_score: True
  lm_config: null

  vocabulary: {{repodir}}/vocabularies/librispeech/wordpiece/train_1000_50.tokens
  vocab_size: 1000
  max_token_length: 50
  max_unique_chars: 1000
  reserved_tokens:
    - "<pad>"
    - "<unk>"
  normalization_form: NFKC
  num_iterations: 4

  train_files: null

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
    metadata: {{repodir}}/vocabularies/librispeech/wordpiece/train_1000_50.metadata.json

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
    warmup_steps: 15000
    max_lr: 0.0025

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

#### Training

```bash
python /path/to/TensorFlowASR/examples/transducer/contextnet/train.py \
    --config-path=/path/to/config.j2 \
    --mxp=strict \
    --jit-compile \
    --tfrecords
```

Outputs:

```
INFO:tensorflow:Use RNNT loss in TensorFlow
INFO:tensorflow:Deallocate tpu buffers before initializing tpu system.
INFO:tensorflow:All TPUs: [LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:0', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:1', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:2', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:3', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:4', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:5', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:6', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:7', device_type='TPU')]
INFO:tensorflow:Found TPU system:
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:USING mixed precision policy mixed_bfloat16
INFO:tensorflow:Loading wordpiece ...
INFO:tensorflow:Loading metadata from /content/TensorFlowASR/vocabularies/librispeech/wordpiece/train_1000_50.metadata.json ...
INFO:tensorflow:TFRecords're already existed: train
INFO:tensorflow:Use GPU/TPU implementation for RNNT loss
Model: "contextnet"
__________________________________________________________________________________________________________________________________________
 Layer (type)                                            Output Shape                                      Param #             Trainable  
==========================================================================================================================================
 encoder (ContextNetEncoder)                             ((8, 372, 640),                                   6888392             Y          
                                                          (8,))                                                                           
                                                                                                                                          
 prediction (TransducerPrediction)                       (8, 203, 512)                                     3002368             Y          
                                                                                                                                          
 joint (TransducerJoint)                                 (8, 372, 203, 1000)                               939496              Y          
                                                                                                                                          
==========================================================================================================================================
Total params: 10,830,258
Trainable params: 10,771,120
Non-trainable params: 59,138
__________________________________________________________________________________________________________________________________________
Epoch 1/300
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
4394/4394 [==============================] - 3321s 659ms/step - loss: 354.1810 - per_batch_avg_loss: 354.1461
Epoch 2/300
4394/4394 [==============================] - 2871s 653ms/step - loss: 208.4284 - per_batch_avg_loss: 208.4139
Epoch 3/300
4394/4394 [==============================] - 2889s 657ms/step - loss: 107.2801 - per_batch_avg_loss: 107.2722
Epoch 4/300
4394/4394 [==============================] - 2892s 658ms/step - loss: 53.1701 - per_batch_avg_loss: 53.1681
Epoch 5/300
4394/4394 [==============================] - 2897s 659ms/step - loss: 36.3960 - per_batch_avg_loss: 36.3937
Epoch 6/300
4394/4394 [==============================] - 2895s 659ms/step - loss: 29.5058 - per_batch_avg_loss: 29.5055
Epoch 7/300
4394/4394 [==============================] - 2899s 660ms/step - loss: 25.5613 - per_batch_avg_loss: 25.5621
Epoch 8/300
4394/4394 [==============================] - 2898s 660ms/step - loss: 22.9065 - per_batch_avg_loss: 22.9076
Epoch 9/300
4394/4394 [==============================] - 2900s 660ms/step - loss: 20.9543 - per_batch_avg_loss: 20.9534
Epoch 10/300
4394/4394 [==============================] - 2894s 659ms/step - loss: 19.4345 - per_batch_avg_loss: 19.4343
Epoch 11/300
4394/4394 [==============================] - 2898s 660ms/step - loss: 18.1739 - per_batch_avg_loss: 18.1751
Epoch 12/300
4394/4394 [==============================] - 2903s 661ms/step - loss: 17.1167 - per_batch_avg_loss: 17.1171
Epoch 13/300
4394/4394 [==============================] - 2899s 660ms/step - loss: 16.2142 - per_batch_avg_loss: 16.2144
Epoch 14/300
4394/4394 [==============================] - 2899s 660ms/step - loss: 15.4079 - per_batch_avg_loss: 15.4081
Epoch 15/300
4394/4394 [==============================] - 2894s 659ms/step - loss: 14.6829 - per_batch_avg_loss: 14.6824
Epoch 16/300
4394/4394 [==============================] - 2897s 659ms/step - loss: 14.0212 - per_batch_avg_loss: 14.0217
Epoch 17/300
4394/4394 [==============================] - 2900s 660ms/step - loss: 13.4270 - per_batch_avg_loss: 13.4276
Epoch 18/300
4394/4394 [==============================] - 2893s 658ms/step - loss: 12.8534 - per_batch_avg_loss: 12.8523
Epoch 19/300
4394/4394 [==============================] - 2904s 661ms/step - loss: 12.3035 - per_batch_avg_loss: 12.3036
Epoch 20/300
4394/4394 [==============================] - 2894s 659ms/step - loss: 11.8388 - per_batch_avg_loss: 11.8391
Epoch 21/300
4394/4394 [==============================] - 2892s 658ms/step - loss: 11.3558 - per_batch_avg_loss: 11.3556
Epoch 22/300
4394/4394 [==============================] - 2890s 658ms/step - loss: 10.8828 - per_batch_avg_loss: 10.8833
Epoch 23/300
4394/4394 [==============================] - 2890s 658ms/step - loss: 10.4597 - per_batch_avg_loss: 10.4601
Epoch 24/300
4394/4394 [==============================] - 2888s 657ms/step - loss: 10.0510 - per_batch_avg_loss: 10.0514
Epoch 25/300
4394/4394 [==============================] - 2893s 658ms/step - loss: 9.6318 - per_batch_avg_loss: 9.6321
Epoch 26/300
4394/4394 [==============================] - 2900s 660ms/step - loss: 9.2559 - per_batch_avg_loss: 9.2558
Epoch 27/300
4394/4394 [==============================] - 2897s 659ms/step - loss: 8.8785 - per_batch_avg_loss: 8.8787
Epoch 28/300
4394/4394 [==============================] - 2898s 659ms/step - loss: 8.5476 - per_batch_avg_loss: 8.5474
Epoch 29/300
4394/4394 [==============================] - 2895s 659ms/step - loss: 8.2002 - per_batch_avg_loss: 8.2000
Epoch 30/300
4394/4394 [==============================] - 3303s 661ms/step - loss: 8.0303 - per_batch_avg_loss: 8.0302
Epoch 31/300
4394/4394 [==============================] - 2874s 654ms/step - loss: 7.5840 - per_batch_avg_loss: 7.5840
Epoch 32/300
4394/4394 [==============================] - 2889s 658ms/step - loss: 7.2728 - per_batch_avg_loss: 7.2727
Epoch 33/300
4394/4394 [==============================] - 2874s 654ms/step - loss: 6.9903 - per_batch_avg_loss: 6.9904
Epoch 34/300
4394/4394 [==============================] - 2883s 656ms/step - loss: 6.7392 - per_batch_avg_loss: 6.7389
Epoch 35/300
4394/4394 [==============================] - 2888s 657ms/step - loss: 6.4757 - per_batch_avg_loss: 6.4754
Epoch 36/300
4394/4394 [==============================] - 2895s 659ms/step - loss: 6.2007 - per_batch_avg_loss: 6.2010
Epoch 37/300
4394/4394 [==============================] - 2892s 658ms/step - loss: 5.9911 - per_batch_avg_loss: 5.9912
Epoch 38/300
4394/4394 [==============================] - 2896s 659ms/step - loss: 5.7801 - per_batch_avg_loss: 5.7799
Epoch 39/300
4394/4394 [==============================] - 2886s 657ms/step - loss: 5.5511 - per_batch_avg_loss: 5.5512
Epoch 40/300
4394/4394 [==============================] - 2893s 658ms/step - loss: 5.3473 - per_batch_avg_loss: 5.3477
Epoch 41/300
4394/4394 [==============================] - 2897s 659ms/step - loss: 5.1490 - per_batch_avg_loss: 5.1488
Epoch 42/300
4394/4394 [==============================] - 3323s 662ms/step - loss: 5.1151 - per_batch_avg_loss: 5.1150
Epoch 43/300
4394/4394 [==============================] - 2868s 653ms/step - loss: 4.7958 - per_batch_avg_loss: 4.7961
Epoch 44/300
4394/4394 [==============================] - 2871s 653ms/step - loss: 4.6019 - per_batch_avg_loss: 4.6019
Epoch 45/300
4394/4394 [==============================] - 2872s 654ms/step - loss: 4.4455 - per_batch_avg_loss: 4.4454
Epoch 46/300
4394/4394 [==============================] - 2880s 655ms/step - loss: 4.2980 - per_batch_avg_loss: 4.2980
Epoch 47/300
4394/4394 [==============================] - 2864s 652ms/step - loss: 4.1674 - per_batch_avg_loss: 4.1674
Epoch 48/300
4394/4394 [==============================] - 2858s 650ms/step - loss: 4.0385 - per_batch_avg_loss: 4.0385
Epoch 49/300
4394/4394 [==============================] - 2860s 651ms/step - loss: 3.9321 - per_batch_avg_loss: 3.9321
Epoch 50/300
4394/4394 [==============================] - 2882s 656ms/step - loss: 3.8276 - per_batch_avg_loss: 3.8274
Epoch 51/300
4394/4394 [==============================] - 2885s 657ms/step - loss: 3.7106 - per_batch_avg_loss: 3.7105
Epoch 52/300
4394/4394 [==============================] - 2876s 655ms/step - loss: 3.6935 - per_batch_avg_loss: 3.6934
Epoch 53/300
4394/4394 [==============================] - 2862s 651ms/step - loss: 3.6176 - per_batch_avg_loss: 3.6176
Epoch 54/300
4394/4394 [==============================] - 2907s 661ms/step - loss: 3.5398 - per_batch_avg_loss: 3.5400
Epoch 55/300
4394/4394 [==============================] - 2880s 655ms/step - loss: 3.4705 - per_batch_avg_loss: 3.4707
Epoch 56/300
4394/4394 [==============================] - 2880s 655ms/step - loss: 3.4127 - per_batch_avg_loss: 3.4126
Epoch 57/300
4394/4394 [==============================] - 2910s 662ms/step - loss: 3.4820 - per_batch_avg_loss: 3.4821
Epoch 58/300
4394/4394 [==============================] - 2890s 658ms/step - loss: 3.5663 - per_batch_avg_loss: 3.5663
Epoch 59/300
4394/4394 [==============================] - 2894s 659ms/step - loss: 3.6664 - per_batch_avg_loss: 3.6661
Epoch 60/300
4394/4394 [==============================] - 2884s 656ms/step - loss: 3.8075 - per_batch_avg_loss: 3.8073
Epoch 61/300
4394/4394 [==============================] - 2869s 653ms/step - loss: 3.9362 - per_batch_avg_loss: 3.9362
Epoch 62/300
4394/4394 [==============================] - 2880s 655ms/step - loss: 4.0172 - per_batch_avg_loss: 4.0174
Epoch 63/300
4394/4394 [==============================] - 2894s 659ms/step - loss: 4.0682 - per_batch_avg_loss: 4.0683
Epoch 64/300
4394/4394 [==============================] - 2919s 664ms/step - loss: 4.0974 - per_batch_avg_loss: 4.0974
Epoch 65/300
4394/4394 [==============================] - 2923s 665ms/step - loss: 4.2700 - per_batch_avg_loss: 4.2701
Epoch 66/300
4394/4394 [==============================] - 2924s 665ms/step - loss: 4.4607 - per_batch_avg_loss: 4.4609
Epoch 67/300
4394/4394 [==============================] - 2918s 664ms/step - loss: 4.5037 - per_batch_avg_loss: 4.5036
Epoch 68/300
4394/4394 [==============================] - 2915s 663ms/step - loss: 4.5902 - per_batch_avg_loss: 4.5899
```

#### Testing

```bash
python /path/to/TensorFlowASR/examples/transducer/contextnet/test.py \
    --config-path=/path/to/config.j2 \
    --saved=/path/to/models/wp1k-contextnet/only-data/checkpoints/40.h5 \
    --output=/path/to/models/wp1k-contextnet/only-data/tests/40.tsv \
    --bs=1
```

RNNT Loss Curves:



Error Rates:

| Dataset    |  Mode  | Batch size | Epoch |      WER (%)       |      CER (%)       |
| :--------- | :----: | :--------: | :---: | :----------------: | :----------------: |
| test-clean | greedy |     1      |  40   | 18.036746978759766 |  8.55042114853859  |
| test-clean | greedy |     1      |  56   | 18.39812844991684  | 8.690726011991501  |
| test-other | greedy |     1      |  56   | 38.31839859485626  | 21.644461154937744 |