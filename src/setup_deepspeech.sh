#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/mozilla/DeepSpeech
#
# If you get the error: fatal error: portaudio.h: No such file or directory
# be sure to install all packages in apt-packages.txt.
set -e

[ -d ../.env_deepspeech ] && rm -Rf ../.env_deepspeech
virtualenv -p python3 ../.env_deepspeech
. ../.env_deepspeech/bin/activate

pip install deepspeech
pip install -r ../requirements.txt

AUDIO_DIR=../data/audio
DATA_DIR=../data/models/deepspeech

rm -Rf $DATA_DIR
[ ! -d $DATA_DIR ] && mkdir -p $DATA_DIR || true
cd $DATA_DIR
# Update this according to the most recent version on the releases page
# https://github.com/mozilla/DeepSpeech/releases
#wget --continue -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz | tar xvfz -
#wget --continue -O deepspeech.tar.gz https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
#wget --continue -O deepspeech.tar.gz https://github.com/mozilla/DeepSpeech/releases/download/v0.2.0/deepspeech-0.2.0-models.tar.gz
#wget --continue -O deepspeech.tar.gz https://github.com/mozilla/DeepSpeech/releases/download/v0.4.0/deepspeech-0.4.0-models.tar.gz
#wget --continue -O deepspeech.tar.gz https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz
wget --continue -O deepspeech.tar.gz https://github.com/mozilla/DeepSpeech/releases/download/v0.6.0/deepspeech-0.6.0-models.tar.gz
[ ! -d models ] && mkdir models || true
tar xvfz deepspeech.tar.gz --strip 1 -C models
cd ../../../src

#chimit deepspeech $DATA_DIR/models/output_graph.pb $AUDIO_DIR/have-a-good-weekend.rate16k-mono.wav $DATA_DIR/models/alphabet.txt $DATA_DIR/models/lm.binary $DATA_DIR/models/trie
time python test_deepspeech.py
