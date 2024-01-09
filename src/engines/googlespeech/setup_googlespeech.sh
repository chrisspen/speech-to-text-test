#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition
set -e

[ -d ../.env_googlespeech ] && rm -Rf ../.env_googlespeech
virtualenv -p python3 ../.env_googlespeech
. ../.env_googlespeech/bin/activate

pip3 install --only-binary scipy -r ../requirements.txt

time python3 test_googlespeech.py
