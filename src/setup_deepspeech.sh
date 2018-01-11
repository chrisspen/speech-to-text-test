#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/mozilla/DeepSpeech

[ ! -d ../.env_deepspeech ] && virtualenv ../.env_deepspeech
. ../.env_deepspeech/bin/activate

pip install deepspeech
pip install -r ../requirements.txt

AUDIO_DIR=../data/audio
DATA_DIR=../data/models/deepspeech

cd $DATA_DIR
#wget --continue -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz | tar xvfz -
wget --continue -O deepspeech.tar.gz https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar xvfz deepspeech.tar.gz
cd ../../../src

#chimit deepspeech $DATA_DIR/models/output_graph.pb $AUDIO_DIR/have-a-good-weekend.rate16k-mono.wav $DATA_DIR/models/alphabet.txt $DATA_DIR/models/lm.binary $DATA_DIR/models/trie
python test_deepspeech.py
