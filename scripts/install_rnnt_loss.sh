#!/usr/bin/env sh

mkdir -p externals
cd ./externals || exit

# Install rnnt_loss
if [ ! -d warp-transducer ]; then
    git clone https://github.com/usimarit/warp-transducer.git
fi

cd ./warp-transducer || exit
rm -rf build
mkdir -p build && cd build || exit

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

make

cd ../tensorflow_binding || exit

if [ "$CUDA_HOME" ]; then
  CUDA="$CUDA_HOME" python setup.py install
else
  python setup.py install
fi

cd ../..

cd ..
