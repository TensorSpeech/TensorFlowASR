#!/usr/bin/env bash

python3 -m pip install -r requirements.text.txt

case "$1" in
tpu)
    python3 -m pip uninstall -y tensorflow
    python3 -m pip install -r requirements.tpu.txt -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
;;
gpu)
    python3 -m pip install -r requirements.gpu.txt
;;
cpu)
    python3 -m pip install -r requirements.cpu.txt
;;
apple)
    python3 -m pip install -r requirements.apple.txt
;;
*) echo -e "Usage: $0 <tpu|gpu|cpu|apple>"
esac

python3 -m pip uninstall -y keras # use keras-nightly
python3 -m pip install -r requirements.txt --force

case "$2" in
dev)
    python3 -m pip install -r requirements.dev.txt
    python3 -m pip install -e .
;;
esac