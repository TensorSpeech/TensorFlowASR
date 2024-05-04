# Testing Tutorial

These commands are example for librispeech dataset, but we can apply similar to other datasets

## 1. Install packages

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

The config file is under format `config.yml.j2` which is jinja2 format with yaml content

Please take a look in some examples for config files in `examples/*/*.yml.j2`

The config file is the same as the config used for training

## 4. [Optional][Required if not exists] Generate vocabulary and metadata

Use the same vocabulary file used in training

```bash
python scripts/prepare_vocab_and_metadata.py \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/datadir
```

The inputs, outputs and other options of vocabulary are defined in the config file

## 5. Run testing

```bash
python examples/test.py \
--config-path /path/to/config.yml.j2 \
--dataset_type slice \
--datadir /path/to/datadir \
--outputdir /path/to/modeldir/tests \
--h5 /path/to/modeldir/weights.h5
## See others params
python examples/test.py --help
```