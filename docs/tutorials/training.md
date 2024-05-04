# Training Tutorial

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
    --directory=/path/to/dataset/train-clean-100 \
    --output=/path/to/dataset/train-clean-100/transcripts.tsv
```

Do the same thing with `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`

For other datasets, you must prepare your own python script like the `scripts/create_librispeech_trans.py`

## 3. Prepare config file

The config file is under format `config.yml.j2` which is jinja2 format with yaml content

Please take a look in some examples for config files in `examples/*/*.yml.j2`

## 4. [Optional][Required if using TPUs] Create tfrecords

```bash
python scripts/create_tfrecords.py \
    --config-path=/path/to/config.yml.j2 \
    --mode=\["train","eval","test"\] \
    --datadir=/path/to/datadir
```

You can reduce the flag `--modes` to `--modes=\["train","eval"\]` to only create train and eval datasets

## 5. Generate vocabulary and metadata

This step requires defining path to vocabulary file and other options for generating vocabulary in config file.

```bash
python scripts/prepare_vocab_and_metadata.py \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/datadir
```

The inputs, outputs and other options of vocabulary are defined in the config file


## 6. Run training

```bash
python examples/train.py \
    --mxp=auto \
    --jit-compile \
    --config-path=/path/to/config.yml.j2 \
    --dataset-type=tfrecord \
    --modeldir=/path/to/modeldir \
    --datadir=/path/to/datadir
## See others params
python examples/train.py --help
```