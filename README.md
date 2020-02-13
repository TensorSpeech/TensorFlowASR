# Vietnamese Automatic Speech Recognition

VASR Implementation in Tensorflow Estimator.

## Dataset

Collected from many sources:

1. Vivos
2. InfoRe (25hr + 500hr)

## Running

Example config file can be found in directory ```configs```.

Run the following commands to install dependencies and run project:

```bash
pip install tensorflow-gpu
pip install -r requirements.txt
python run.py --mode=train
```

Run the following command to see the flags:

```bash
python run.py --helpfull
```

## References

1. [https://github.com/tensorflow/models/tree/master/research/deep_speech](https://github.com/tensorflow/models/tree/master/research/deep_speech)
2. [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595)
