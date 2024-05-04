#!/usr/bin/env bash

PROJECT_DIR=$(realpath "$(dirname $0)/..")
cd "$PROJECT_DIR" || exit

mkdir -p $PROJECT_DIR/externals
cd $PROJECT_DIR/externals || exit

TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")

# Install rnnt_loss
if [ ! -d warp-transducer ]; then
    git clone --depth 1 https://github.com/nglehuy/warp-transducer.git
    cd $PROJECT_DIR/externals/warp-transducer/tensorflow_binding
    git clone --depth 1 --branch v$TF_VERSION https://github.com/tensorflow/tensorflow.git
    cd ../../
fi

TENSORFLOW_SRC_PATH="$PROJECT_DIR/externals/warp-transducer/tensorflow_binding/tensorflow"

rm -rf $PROJECT_DIR/externals/warp-transducer/build
mkdir -p $PROJECT_DIR/externals/warp-transducer/build
cd $PROJECT_DIR/externals/warp-transducer/build || exit

if [ "$CUDA_HOME" ]; then
  cmake \
      -DUSE_NAIVE_KERNEL=OFF \
      -DWITH_GPU=ON \
      -DCMAKE_C_COMPILER_LAUNCHER="$(which gcc)" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="$(which g++)"  \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" ..
else
  cmake \
      -DUSE_NAIVE_KERNEL=OFF \
      -DWITH_GPU=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER="$(which gcc)" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="$(which g++)" ..
fi

make -j $(nproc)

cd $PROJECT_DIR/externals/warp-transducer/tensorflow_binding || exit

if [ "$CUDA_HOME" ]; then
  CUDA="$CUDA_HOME" python3 setup.py install
else
  python3 setup.py install
fi

cd $PROJECT_DIR || exit