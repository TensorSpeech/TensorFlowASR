#!/usr/bin/env bash

PROJECT_DIR=$(realpath "$(dirname $0)/..")

mkdir -p $PROJECT_DIR/externals
cd $PROJECT_DIR/externals || exit

# Install baidu's beamsearch_with_lm
if [ ! -d ctc_decoders ]; then
    git clone --depth 1 https://github.com/nglehuy/ctc_decoders.git
    cd ./ctc_decoders || exit
    chmod a+x setup.sh
    ./setup.sh
fi

cd $PROJECT_DIR || exit
