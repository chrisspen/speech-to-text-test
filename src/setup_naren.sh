#!/bin/bash
# CS 2019.11.21
set -e

#[ -d ../.env_naren ] && rm -Rf ../.env_naren
#virtualenv -p python3.7 ../.env_naren
#. ../.env_naren/bin/activate

. /home/chris/git/deepspeech.pytorch/.env/bin/activate

#pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
#pip install numpy scipy h5py tensorflow==1.13.1
#pip install PyYAML

time python test_naren.py
