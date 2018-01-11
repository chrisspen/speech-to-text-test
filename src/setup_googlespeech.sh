#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition

[ ! -d ../.env_googlespeech ] && virtualenv ../.env_googlespeech
. ../.env_googlespeech/bin/activate

pip install SpeechRecognition
pip install --only-binary scipy -r ../requirements.txt

python test_googlespeech.py
