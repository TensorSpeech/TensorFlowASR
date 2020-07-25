# Dataset Structures :kissing:

## ASR Datasets

An ASR dataset is some `.tsv` files in format: `PATH\tDURATION\tTRANSCRIPT`. You must create those files by your own with your own data and methods.

**Note**: Each `.tsv` file must include a header `PATH\tDURATION\tTRANSCRIPT` because it will remove these headers when loading dataset, otherwise you will lose 1 data file :sob:

**For transcript**, if you want to include characters such as dots, commas, double quote, etc.. you must create your own `.txt` vocabulary file. Default is [English](../featurizers/english.txt)

**Inputs**

```python
class ASRTFRecordDataset(ASRDataset):
    """ Dataset for ASR using TFRecords """
    def __init__(self,
                 data_paths: list,
                 tfrecords_dir: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 stage: str,
                 augmentations: dict = None,
                 shuffle: bool = False)

class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """
    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: dict = None,
                 shuffle: bool = False)
```

**Outputs when iterating**

```python
(path, features, input_lengths, labels, label_lengths, pred_inp)
```

## SEGAN Datasets

A SEGAN Dataset is some path to **clean** `.wav` files and a path to **noise** `.wav` files. While training, it will add noises from noisy audio files into clean audio signals _on the fly_ according to your configuration :kissing_smiling_eyes:

**Inputs**

```python
class SeganDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 data_paths: list,
                 noises_config: dict,
                 speech_config: dict,
                 shuffle: bool = False)
```

**Outputs when iterating for training**

```python
(clean_wav_slices, noisy_wav_slices)
```

**Outputs when iterating for testing**

```python
(path, clean_wav, noisy_wav_slices)
```
