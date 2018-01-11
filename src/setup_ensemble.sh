#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition

[ ! -d ../.env_ensemble ] && virtualenv ../.env_ensemble
. ../.env_ensemble/bin/activate

sudo apt-get install swig libpulse-dev

pip install --only-binary scipy pocketsphinx SpeechRecognition deepspeech
pip install --only-binary scipy -r ../requirements.txt

python test_ensemble.py
