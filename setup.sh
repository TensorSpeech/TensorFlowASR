#!/usr/bin/env sh

pip install -r requirements.txt

if [ ! -d beamsearch_with_lm ]; then
    git clone https://github.com/huylenguyen806/beamsearch_with_lm.git
    
    cd ./beamsearch_with_lm
    
    chmod a+x setup.sh
    chown $USER:$USER setup.sh
    ./setup.sh
    
    cd ..
fi

if [ ! -d warp-transducer ]; then
    git clone https://github.com/noahchalifour/warp-transducer.git
    
    cd ./warp-transducer
    
    mkdir build
    cd build
    cmake ..
    make
    
    cd ../tensorflow_binding
    python setup.py install
    
    cd ../..
fi

if [ ! -d semetrics ]; then
    git clone https://github.com/usimarit/semetrics

    cd semetrics
    chmod a+x setup.sh
    chown $USER:$USER setup.sh
    ./setup.sh

    cd ..
fi
