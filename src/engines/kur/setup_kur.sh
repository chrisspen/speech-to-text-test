#!/bin/bash
# CS 2017.1.7
# Sets up the program deepgram.
# Based on instructions at:
#
#   https://blog.deepgram.com/how-to-train-baidus-deepspeech-model-with-kur/
#   https://kur.deepgram.com/
#
set -e

[ -d ../.env_kur ] && rm -Rf ../.env_kur
virtualenv -p python3.7 ../.env_kur
. ../.env_kur/bin/activate

pip install kur tensorflow==1.15.0 keras==2.2.4
pip install -r ../requirements.txt

cd /home/chris/git/kur/examples
time kur -v train speech.yml
