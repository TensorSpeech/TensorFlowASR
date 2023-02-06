# Training Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Install packages (tf>=2.8)

If you use google colab, it's recommended to use the tensorflow version pre-installed on the colab itself

```bash
pip install -e ".[tf2.x]"
```

## 2. Prepare transcripts files

This is the example for preparing transcript files for librispeech data corpus

```bash
python scripts/create_librispeech_trans.py \
--directory=/path/to/dataset/train-clean-100 \
--output=/path/to/dataset/train-clean-100/transcripts.tsv
```

Do the same thing with `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`

For other datasets, you must prepare your own python script like the `scripts/create_librispeech_trans.py`

## 3. Prepare config file

The config file is under format `config.j2` which is jinja2 format

Please take a look in some examples for config files in `examples/*/config*.j2`

## 4. [Optional][Required if using TPUs] Create tfrecords

```bash
python scripts/create_tfrecords.py \
--mode=train \
--config-path=/path/to/config.j2 \
--tfrecords-dir=/path/to/dataset/tfrecords \
--tfrecords-shards=16 \ # available options are from 1 -> inf
--shuffle \
/path/to/dataset/train-clean-100/transcripts.tsv \
/path/to/dataset/train-clean-360/transcripts.tsv \
/path/to/dataset/train-other-500/transcripts.tsv
```

Reduce the `--tfrecords-shards` if the size of the dataset is small

Do the same thing with `--mode=eval` and `--mode=test` if needed, corresponds to `dev` and `test` datasets

## 5. Generate vocabulary


Characters:

```bash
Prepare like the files in vocabularies/*.characters
```

Wordpiece:

```bash
python scripts/generate_vocab_wordpiece.py --config-path=/path/to/config.j2
```

Sentencepiece:

```bash
python scripts/generate_vocab_sentencepiece.py --config-path=/path/to/config.j2
```

The inputs, outputs and other options of vocabulary are defined in the config file

## 5. [Optional][Required if using TPUs] Generate metadata.json

The metadata json file contains all the metadata of dataset derived with the current config of `speech_config` and `decoder_config` in the config file

These metadata is for **static-shape** training, which is required for TPUs

Static shape means that it will pad each record to the longest record size of the whole data, therefore if you use with `train` mode and `eval` mode, you have to generate metadata for both stages (aka modes) so that when loading the dataset, it will get the longest record size from both train and eval modes

```bash
python scripts/generate_metadata.py \
--stage=train \
--config-path=/path/to/config.j2 \
--metadata=/path/to/metadata.json \
/path/to/dataset/train-clean-100/transcripts.tsv \
/path/to/dataset/train-clean-360/transcripts.tsv \
/path/to/dataset/train-other-500/transcripts.tsv
# same thing with eval mode
python scripts/generate_metadata.py \
--stage=eval \
--config-path=/path/to/config.j2 \
--metadata=/path/to/metadata.json \
/path/to/dataset/dev-clean/transcripts.tsv \
/path/to/dataset/dev-other/transcripts.tsv
```

## 6. Update config file

Update config file with:
-  The paths to transcript files (and tfrecords if used)
-  The path to metadata json file (if use static shape training)

## 7. Run training

```bash
python examples/conformer/train.py --mxp --jit-compile --config-path=/path/to/config.j2 --tfrecords
```

See other options for each example