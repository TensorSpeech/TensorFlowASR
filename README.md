<h1 align="center">
<p>TiramisuASR :cake:</p>
<p align="center">
<a href="https://github.com/usimarit/TiramisuASR/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/usimarit/TiramisuASR">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
<img alt="ubuntu" src="https://img.shields.io/badge/ubuntu-%3E%3D18.04-yellowgreen">
</p>
</h1>
<h2 align="center">
<p>The Newest Automatic Speech Recognition in Tensorflow 2</p>
</h2>

<p align="center">
TiramisuASR implements some speech recognition and speech enhancement architectures such as CTC-based models (Deep Speech 2, etc.), Speech Enhancement Generative Adversarial Network (SEGAN), RNN Transducer (Conformer, etc.). These models can be converted to TFLite to reduce memory and computation for deployment :smile:
</p>

## What's New?

-   Use **nlpaug** as augmentation for speech recognition and enhancement

## :yum: Supported Models

-   **CTCModel** (End2end models using CTC Loss for training)
-   **SEGAN** (Refer to [https://github.com/santi-pdp/segan](https://github.com/santi-pdp/segan)), see [examples/segan](./examples/segan)
-   **Transducer Models** (End2end models using RNNT Loss for training)
-   **Conformer Transducer** (Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100))
    See [examples/conformer](./examples/conformer)

## Requirements

-   Ubuntu distribution (`ctc-decoders` and `semetrics` require some packages from apt)
-   Python 3.6+
-   Tensorflow 2.2+: `pip install tensorflow`

## Setup Environment and Datasets

For _training, testing and using_ **CTC Models**, run `./scripts/install_ctc_decoders.sh`

For _training_ **Transducer Models**, run `./scripts/install_rnnt_loss.sh`

For _testing_ **Speech Enhancement Model** (i.e SEGAN), run `./scripts/install_semetrics.sh`

Install tensorflow: `pip3 install tensorflow` or `pip3 install tf-nightly` (for using tflite)

Install packages: `python3 setup.py install`

Clean up: `python3 setup.py clean --all` (this will remove `/build` contents)

For **setting up datasets**, see [datasets](./tiramisu_asr/datasets/README.md)

## TFLite Convertion

After converting to tflite, the tflite model is like a function that transforms directly from an **audio signal** to **unicode code points**, then we can convert unicode points to string.

1. Install `tf-nightly` using `pip install tf-nightly`
2. Build a model with the same architecture as the trained model, then load the weights from trained model to the built model
3. Load `TFSpeechFeaturizer` and `TextFeaturizer` to model using function `add_featurizers`
4. Convert `recognize_tflite` or `recognize_beam_tflite` function to tflite as follows:

```python
# concrete_func = model.recognize_tflite.get_concrete_function()
concrete_func = model.recognize_beam_tflite.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func]
)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
```

5. Save the converted tflite model as follows:

```python
if not os.path.exists(os.path.dirname(tflite_path)):
    os.makedirs(os.path.dirname(tflite_path))
with open(tflite_path, "wb") as tflite_out:
    tflite_out.write(tflite_model)
```

5. Then the `.tflite` model is ready to be deployed

## Features Extraction

See [features_extraction](./tiramisu_asr/featurizers/README.md)

## Augmentations

See [augmentations](./tiramisu_asr/augmentations/README.md)

## Training & Testing

**Example YAML Config Structure**

```yaml
speech_config: ...
model_config: ...
decoder_config: ...
learning_config:
    augmentations: ...
    dataset_config:
        train_paths: ...
        eval_paths: ...
        test_paths: ...
        tfrecords_dir: ...
    optimizer_config: ...
    running_config:
        batch_size: 8
        num_epochs: 20
        outdir: ...
        log_interval_steps: 500
```

See [examples](./examples/) for some predefined ASR models.

## Collected Datasets

<details><summary>expand</summary><div>

### Vietnamese

1. VIVOS: 15hrs from [here](https://ailab.hcmus.edu.vn/vivos)
2. InfoRe Technology 1: 25hrs Single Person [here](https://files.huylenguyen.com/datasets/infore/25hours.zip)
3. InfoRe Technology 2 (also used in VLSP2019): ~415hrs from [here](https://files.huylenguyen.com/datasets/infore/audiobooks.zip)

</div></details>

## References & Credits

1. [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
2. [NVIDIA OpenSeq2Seq Toolkit](https://github.com/NVIDIA/OpenSeq2Seq)
3. [https://github.com/santi-pdp/segan](https://github.com/santi-pdp/segan)
4. [SEGAN: Speech Enhancement Generative Adversarial Network](https://arxiv.org/abs/1703.09452)
5. [https://github.com/noahchalifour/warp-transducer](https://github.com/noahchalifour/warp-transducer)
6. [Sequence Transduction with Recurrent Neural Network](https://arxiv.org/abs/1211.3711)
