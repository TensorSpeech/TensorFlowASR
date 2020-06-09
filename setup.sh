#!/usr/bin/env bash

mkdir externals || return
cd ./externals || exit

# Install baidu's beamsearch_with_lm
if [ ! -d beamsearch_with_lm ]; then
    git clone https://github.com/huylenguyen806/beamsearch_with_lm.git

    cd ./beamsearch_with_lm || exit
    chmod a+x setup.sh
    chown "$USER":"$USER" setup.sh
    ./setup.sh

    cd ..
fi


# Install rnnt_loss
if [ ! -d warp-transducer ]; then
    git clone https://github.com/noahchalifour/warp-transducer.git

    cd ./warp-transducer || exit
    mkdir build && cd build || exit
    cmake ..
    make

    cd ../tensorflow_binding || exit
    python setup.py install

    cd ../..
fi


if [ ! -d semetrics ]; then
    git clone https://github.com/usimarit/semetrics

    cd semetrics || exit
    python setup.py install

    cd ..

fi

cd ..

python setup.py install
