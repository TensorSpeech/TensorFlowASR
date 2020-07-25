#!/usr/bin/env bash

mkdir externals
cd ./externals || exit

if [ ! -d semetrics ]; then
    git clone https://github.com/usimarit/semetrics.git

    cd semetrics || exit
    pip3 install Cython
    pip3 install oct2py pesq
    python3 setup.py install

    cd ..

fi

cd ..
