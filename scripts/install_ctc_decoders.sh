#!/usr/bin/env bash

mkdir externals
cd ./externals || exit

# Install baidu's beamsearch_with_lm
if [ ! -d ctc_decoders ]; then
    git clone https://github.com/usimarit/ctc_decoders.git

    cd ./ctc_decoders || exit
    chmod a+x setup.sh
    chown "$USER":"$USER" setup.sh
    ./setup.sh

    cd ..
fi

cd ..
