# Dataset Structures :kissing:

To make a custom dataset, inherit the `BaseDataset` class and override following methods:

1. `create` to create `tf.data.Dataset` instance.
2. `parse` for transforming `tf.data.Dataset` during creation by applyting `tf.data.Dataset.map` function.

_Note_: To create transcripts for **librispeech**, see [create_librispeech_trans.py](../../scripts/create_librispeech_trans.py)

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
                 cache: bool = False,
                 shuffle: bool = False)

class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """
    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: dict = None,
                 cache: bool = False,
                 shuffle: bool = False)
```

**Outputs when iterating in train step**

```python
(path, features, input_lengths, labels, label_lengths, pred_inp)
```

**Outputs when iterating in test step**

```python
(path, signals, labels)
```