# Vietnamese (Or other languages) Automatic Speech Recognition and SEGAN

VASR and SEGAN Implementation in Tensorflow Keras.

## Dataset for Vietnamese

Collected from many sources:

1. Vivos: 15hrs from [ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos)
2. InfoRe: 25hrs Single Person + ~415hrs from [facebook](https://www.facebook.com/groups/j2team.community/permalink/1010834009248719/)
3. VLSP: ~415hrs (the same as InfoRe 415hrs)

## Featurizers

### Speech Features

**Speech features** are extracted from the **Signal** with ```sample_rate```, ```frame_ms```, ```stride_ms``` and ```num_feature_bins```.

Speech features has the shape ```(batch, time, num_feature_bins, channels)``` and it contains from 1-4 channels:

1. Spectrogram, Log Mel Spectrogram (log filter banks) or MFCCs
2. Delta features: ```librosa.feature.delta``` from the features extracted on channel 1.
3. Delta deltas features: ```librosa.feature.delta``` with ```order=2``` from the features extracted on channel 1.
4. Pitch features: ```librosa.core.piptrack``` from the signal

### Text Features

**Text features** are read as index from the file ```data/vocabulary.txt``` plus 1 at the end for the blank index. You can extend the characters to match with your language in this file.

## Models

There're 3 main models in this repo: CTCModel, SEGAN and RNNTransducer

**CTCModel** uses DeepSpeech2 as the base model. You can write your custom model instead of DeepSpeech2, make sure that the input matches the speech features and output is in the shape ```(batch, time, num_classes)```.

**SEGAN** was created exactly as the segan repo in the references and tested.

**RNNTransducer** is under development.

## Training

There're 2 training methods for the ASR:

1. Train using ```tf.GradientTape``` with ```tf.data.Dataset.from_generator```.
2. Train using keras built-in function ```fit``` with ```tf.data.TFRecordDataset```.

```tf.GradientTape``` run with ```TFRecords``` causes RAM OOM, so I use ```tf.data.Dataset.from_generator``` instead (slower because it has to read many wav files)

## Evaluation Metrics

*Word Error Rate (WER)* and *Character Error Rate (CER)* are used.

## Running

Example config file can be found in directory ```configs```.

```bash
chmod a+x setup.sh && chown $USER:$USER setup.sh && ./setup.sh # Install dependencies
python $SCRIPT --help # Where $SCRIPT is one of the run_*.py files, --help to see the flags
```

## Tensorflow Lite

Install ```tf-nightly``` to use tflite (due to error ```Quantize weights transformation failed``` in tensorflow 2.2.0)

## References

1. [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595)
2. [https://github.com/NVIDIA/OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq)
3. [https://github.com/santi-pdp/segan](https://github.com/santi-pdp/segan)
