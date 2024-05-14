# TFLite Conversion Tutorial

## Run

```bash
python examples/train.py \
    --config-path=/path/to/config.yml.j2 \
    --h5=/path/to/weight.h5 \
    --output=/path/to/output.tflite
## See others params
python examples/tflite.py --help
```