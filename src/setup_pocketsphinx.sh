#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition

[ ! -d ../.env_pocketsphinx ] && virtualenv ../.env_pocketsphinx
. ../.env_pocketsphinx/bin/activate

sudo apt-get install swig libpulse-dev

pip install pocketsphinx SpeechRecognition
pip install -r ../requirements.txt

python test_pocketsphinx.py
