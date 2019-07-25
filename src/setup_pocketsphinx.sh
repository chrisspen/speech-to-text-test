#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition
set -e

[ -d ../.env_pocketsphinx ] && rm -Rf ../.env_pocketsphinx
virtualenv -p python3 ../.env_pocketsphinx
. ../.env_pocketsphinx/bin/activate

#sudo apt-get install swig libpulse-dev

pip3 install pocketsphinx SpeechRecognition
pip3 install -r ../requirements.txt

time python3 test_pocketsphinx.py
