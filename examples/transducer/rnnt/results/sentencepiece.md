# Sentencepiece RNN Transducer

- [Sentencepiece RNN Transducer](#sentencepiece-rnn-transducer)
  - [LibriSpeech Only Data](#librispeech-only-data)
      - [Config](#config)
      - [Training](#training)
      - [Testing](#testing)


## LibriSpeech Only Data

#### Config

```python
config = """
{% set repodir = "/path/to/TensorFlowASR" %}
{% set modeldir = "/path/to/models/sp1k-rnnt/only-data" %}
{% set datadir = "/path/to/librispeech/tfrecords" %}

model_config:
  name: rnnt
  encoder_reductions:
    0: 4
    1: 2
  encoder_dmodel: 256
  encoder_rnn_type: lstm
  encoder_rnn_units: 512
  encoder_rnn_unroll: False
  encoder_nlayers: 8
  encoder_layer_norm: True
  prediction_label_encode_mode: embedding
  prediction_embed_dim: 512
  prediction_num_rnns: 2
  prediction_rnn_units: 512
  prediction_rnn_type: lstm
  prediction_rnn_unroll: False
  prediction_layer_norm: True
  prediction_projection_units: 256
  joint_dim: 256
  prejoint_encoder_linear: False
  prejoint_prediction_linear: False
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
    enabled: False
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
    enabled: True
    data_paths:
      - {{datadir}}/test-clean/transcripts.tsv
    tfrecords_dir: null
    shuffle: False
    cache: True
    buffer_size: 100
    drop_remainder: False
    stage: test

  optimizer_config:
    beta_1: 0.9
    beta_2: 0.98
    epsilon: 1e-9

  learning_rate_config:
    warmup_steps: 10000
    max_lr_numerator: 0.05

  running_config:
    batch_size: 6
    num_epochs: 300
    checkpoint:
      filepath: {{modeldir}}/checkpoints/{epoch:02d}.h5
      save_best_only: False
      save_weights_only: True
      save_freq: epoch
    backup_and_restore:
      backup_dir: {{modeldir}}/states
    tensorboard:
      log_dir: {{modeldir}}/tensorboard
      write_graph: False
      write_images: False
      update_freq: epoch
      profile_batch: 100
"""
with open("/path/to/config.j2", "w") as file:
    file.write(config)
```

#### Training

```bash
python /path/to/TensorFlowASR/examples/transducer/rnnt/train.py \
    --config-path=/path/to/config.j2 \
    --mxp=strict \
    --jit-compile \
    --tfrecords
```

Outputs:

```
2023-02-17 15:05:56.437429: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:267] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
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
INFO:tensorflow:Loading SentencePiece model ...
INFO:tensorflow:Loading metadata from /content/TensorFlowASR/vocabularies/librispeech/sentencepiece/train_bpe_1000.metadata.json ...
INFO:tensorflow:TFRecords're already existed: train
INFO:tensorflow:Use GPU/TPU implementation for RNNT loss
Model: "rnnt"
__________________________________________________________________________________________________________________________________________
 Layer (type)                                            Output Shape                                      Param #             Trainable  
==========================================================================================================================================
 encoder (RnnTransducerEncoder)                          ((6, 372, 256),                                   13821952            Y          
                                                          (6,))                                                                           
                                                                                                                                          
 prediction (TransducerPrediction)                       (6, 232, 256)                                     4450816             Y          
                                                                                                                                          
 joint (TransducerJoint)                                 (6, 372, 232, 1000)                               257000              Y          
                                                                                                                                          
==========================================================================================================================================
Total params: 18,529,770
Trainable params: 18,529,768
Non-trainable params: 2
__________________________________________________________________________________________________________________________________________
Epoch 1/300
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
5859/5859 [==============================] - 5698s 959ms/step - loss: 202.8960 - avg_loss: 201.8269 - avg_loss_scaled: 25.2734
Epoch 2/300
5859/5859 [==============================] - 5587s 953ms/step - loss: 45.2534 - avg_loss: 45.2210 - avg_loss_scaled: 5.6632
Epoch 3/300
5859/5859 [==============================] - 5590s 954ms/step - loss: 32.3394 - avg_loss: 32.4127 - avg_loss_scaled: 4.0592
Epoch 4/300
5859/5859 [==============================] - 5587s 954ms/step - loss: 25.4292 - avg_loss: 25.5215 - avg_loss_scaled: 3.1960
Epoch 5/300
5859/5859 [==============================] - 5585s 953ms/step - loss: 21.2849 - avg_loss: 21.3877 - avg_loss_scaled: 2.6787
Epoch 6/300
5859/5859 [==============================] - 5587s 954ms/step - loss: 18.3444 - avg_loss: 18.3039 - avg_loss_scaled: 2.2923
Epoch 7/300
5859/5859 [==============================] - 5587s 954ms/step - loss: 16.1143 - avg_loss: 16.1171 - avg_loss_scaled: 2.0184
Epoch 8/300
5859/5859 [==============================] - 5588s 954ms/step - loss: 14.2889 - avg_loss: 14.3153 - avg_loss_scaled: 1.7928
Epoch 9/300
5859/5859 [==============================] - 5591s 954ms/step - loss: 12.7466 - avg_loss: 12.7925 - avg_loss_scaled: 1.6021
Epoch 10/300
5859/5859 [==============================] - 5589s 954ms/step - loss: 11.4285 - avg_loss: 11.4078 - avg_loss_scaled: 1.4286
Epoch 11/300
5859/5859 [==============================] - 5588s 954ms/step - loss: 10.2693 - avg_loss: 10.2889 - avg_loss_scaled: 1.2885
Epoch 12/300
5859/5859 [==============================] - 5588s 954ms/step - loss: 9.2380 - avg_loss: 9.2959 - avg_loss_scaled: 1.1642
Epoch 13/300
5859/5859 [==============================] - 5589s 954ms/step - loss: 8.3355 - avg_loss: 8.3695 - avg_loss_scaled: 1.0481
Epoch 14/300
5859/5859 [==============================] - 5591s 954ms/step - loss: 7.5065 - avg_loss: 7.4994 - avg_loss_scaled: 0.9392
Epoch 15/300
5859/5859 [==============================] - 5592s 954ms/step - loss: 6.7857 - avg_loss: 6.7983 - avg_loss_scaled: 0.8514
Epoch 16/300
5859/5859 [==============================] - 5704s 961ms/step - loss: 6.1724 - avg_loss: 6.2067 - avg_loss_scaled: 0.7773
Epoch 17/300
5859/5859 [==============================] - 5610s 957ms/step - loss: 5.5486 - avg_loss: 5.5334 - avg_loss_scaled: 0.6930
Epoch 18/300
5859/5859 [==============================] - 5601s 956ms/step - loss: 5.0254 - avg_loss: 4.9964 - avg_loss_scaled: 0.6257
Epoch 19/300
5859/5859 [==============================] - 5596s 955ms/step - loss: 4.5515 - avg_loss: 4.5450 - avg_loss_scaled: 0.5692
Epoch 20/300
5859/5859 [==============================] - 5600s 956ms/step - loss: 4.1555 - avg_loss: 4.1187 - avg_loss_scaled: 0.5158
Epoch 21/300
5859/5859 [==============================] - 5593s 955ms/step - loss: 3.7699 - avg_loss: 3.7569 - avg_loss_scaled: 0.4705
Epoch 22/300
5859/5859 [==============================] - 5586s 953ms/step - loss: 3.4460 - avg_loss: 3.4470 - avg_loss_scaled: 0.4317
Epoch 23/300
5859/5859 [==============================] - 5585s 953ms/step - loss: 3.1594 - avg_loss: 3.1790 - avg_loss_scaled: 0.3981
Epoch 24/300
5859/5859 [==============================] - 5592s 954ms/step - loss: 2.8957 - avg_loss: 2.9069 - avg_loss_scaled: 0.3640
Epoch 25/300
5859/5859 [==============================] - 5596s 955ms/step - loss: 2.6875 - avg_loss: 2.6777 - avg_loss_scaled: 0.3353
Epoch 26/300
5859/5859 [==============================] - 5587s 954ms/step - loss: 2.4714 - avg_loss: 2.4769 - avg_loss_scaled: 0.3102
Epoch 27/300
5859/5859 [==============================] - 5583s 953ms/step - loss: 2.2982 - avg_loss: 2.3308 - avg_loss_scaled: 0.2919
Epoch 28/300
5859/5859 [==============================] - 5582s 953ms/step - loss: 2.1336 - avg_loss: 2.1169 - avg_loss_scaled: 0.2651
Epoch 29/300
5859/5859 [==============================] - 5583s 953ms/step - loss: 2.0019 - avg_loss: 2.0183 - avg_loss_scaled: 0.2528
Epoch 30/300
5859/5859 [==============================] - 5584s 953ms/step - loss: 1.8734 - avg_loss: 1.8510 - avg_loss_scaled: 0.2318
```

#### Testing

```bash
python /path/to/TensorFlowASR/examples/transducer/rnnt/test.py \
    --config-path=/path/to/config.j2 \
    --saved=/path/to/models/sp1k-rnnt/only-data/checkpoints/30.h5 \
    --output=/path/to/models/sp1k-rnnt/only-data/tests/30.tsv \
    --bs=1
```

RNNT Loss Curves:



Error Rates:

| Dataset    |  Mode  | Batch size | Epoch |      WER (%)      |      CER (%)       |
| :--------- | :----: | :--------: | :---: | :---------------: | :----------------: |
| test-clean | greedy |     1      |  30   | 14.17757123708725 | 6.1642616987228394 |
| test-other | greedy |     1      |  30   | 33.20023715496063 | 17.79550015926361  |