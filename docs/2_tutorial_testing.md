# Testing Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Install packages (tf>=2.8)

If you use google colab, it's recommended to use the tensorflow version pre-installed on the colab itself

```bash
pip uninstall -y TensorFlowASR # uninstall for clean install if needed
pip install ".[tf2.x]"
```

## 2. Prepare transcripts files

This is the example for preparing transcript files for librispeech data corpus

```bash
python scripts/create_librispeech_trans.py \
--directory=/path/to/dataset/test-clean \
--output=/path/to/dataset/test-clean/transcripts.tsv
```

Do the same thing with `test-clean`, `test-other`

For other datasets, you must prepare your own python script like the `scripts/create_librispeech_trans.py`

## 3. Prepare config file

The config file is under format `config.j2` which is jinja2 format

Please take a look in some examples for config files in `examples/*/config*.j2`

The config file is the same as the config used for training

## 4. [Optional][Required if not exists] Generate vocabulary

Use the same vocabulary file used in training

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

## 5. Update config file

Update config file with:
-  The paths to transcript files for test stage

## 6. Run testing

```bash
python examples/conformer/test.py --config-path=/path/to/config.j2 --saved=/path/to/saved_weights.h5 --bs=1 --output=/path/to/test.tsv
```

See other options for each example