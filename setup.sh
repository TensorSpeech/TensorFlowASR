#!/usr/bin/env sh

pip install -r requirements.txt

git clone https://github.com/huylenguyen806/beamsearch_with_lm.git

cd ./beamsearch_with_lm

chmod a+x setup.sh
chown $USER:$USER setup.sh

./setup.sh

