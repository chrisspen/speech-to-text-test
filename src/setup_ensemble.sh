#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition

[ -d ../.env_ensemble ] && rm -Rf ../.env_ensemble
virtualenv -p python3 ../.env_ensemble
. ../.env_ensemble/bin/activate

#sudo apt-get install swig libpulse-dev

pip3 install --only-binary scipy pocketsphinx deepspeech
pip3 install --only-binary scipy -r ../requirements.txt

python3 test_ensemble.py
