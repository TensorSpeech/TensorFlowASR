#!/usr/bin/env bash

PROJECT_DIR=$(realpath "$(dirname $0)/..")
cd "$PROJECT_DIR" || exit

mkdir -p $PROJECT_DIR/externals
cd $PROJECT_DIR/externals || exit

TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")

# Install rnnt_loss
if [ ! -d warp-ctc ]; then
    git clone --depth 1 https://github.com/nglehuy/warp-ctc.git
    cd $PROJECT_DIR/externals/warp-ctc/tensorflow_binding
    if [ ! -d tensorflow ]; then
        git clone --depth 1 --branch v$TF_VERSION https://github.com/tensorflow/tensorflow.git
    fi
    cd ../../
fi

export TENSORFLOW_SRC_PATH="$PROJECT_DIR/externals/warp-ctc/tensorflow_binding/tensorflow"

rm -rf $PROJECT_DIR/externals/warp-ctc/build
mkdir -p $PROJECT_DIR/externals/warp-ctc/build
cd $PROJECT_DIR/externals/warp-ctc/build || exit

if [ "$CUDA_HOME" ]; then
  cmake \
      -DWITH_GPU=ON \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" ..
else
  cmake \
      -DWITH_GPU=OFF \
      ..
fi

make -j $(nproc)

cd $PROJECT_DIR/externals/warp-ctc/tensorflow_binding || exit

if [ "$CUDA_HOME" ]; then
  CUDA="$CUDA_HOME" python3 setup.py install
else
  python3 setup.py install
fi

cd $PROJECT_DIR || exit