#!/bin/bash
# CS 2019.11.21
set -e

[ -d ../.env_kaldi ] && rm -Rf ../.env_kaldi
virtualenv -p python3.7 ../.env_kaldi
. ../.env_kaldi/bin/activate

sudo apt install libatlas-base-dev
sudo ln -s /usr/include/x86_64-linux-gnu/atlas /usr/include/atlas

pip install Cython
pip install numpy py-kaldi-asr PyYAML
#pip install kaldi tensorflow==1.15.0 keras==2.2.4
#pip install -r ../requirements.txt

time python test_kaldi.py
