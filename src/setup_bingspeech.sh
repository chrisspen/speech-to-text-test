#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/mozilla/DeepSpeech

[ ! -d ../.env_bingspeech ] && virtualenv ../.env_bingspeech
. ../.env_bingspeech/bin/activate

pip install SpeechRecognition
pip install --only-binary scipy -r ../requirements.txt

python test_bingspeech.py
