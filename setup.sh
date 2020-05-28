#!/usr/bin/env sh

sudo apt install libboost-all-dev swig sox
sudo snap install octave

pip install -r requirements.txt

git clone https://github.com/huylenguyen806/beamsearch_with_lm.git

cd ./beamsearch_with_lm

chmod a+x setup.sh
chown $USER:$USER setup.sh

./setup.sh

cd ..

git clone https://github.com/noahchalifour/warp-transducer.git

cd ./warp-transducer

mkdir build
cd build
cmake ..
make

cd ../tensorflow_binding
python setup.py install

cd ../..

git clone https://github.com/usimarit/semetrics


