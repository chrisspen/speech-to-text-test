#!/bin/bash
# CS 2019.11.21
set -e

[ -d ../.env_wav2letter ] && rm -Rf ../.env_wav2letter
virtualenv -p python3.7 ../.env_wav2letter
. ../.env_wav2letter/bin/activate

pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy scipy h5py tensorflow==1.13.1
pip install PyYAML

time python test_wav2letter.py
