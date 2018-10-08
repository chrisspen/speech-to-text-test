#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition
set -e

[ -d ../.env_houndify ] && rm -Rf ../.env_houndify
virtualenv -p python3 ../.env_houndify
. ../.env_houndify/bin/activate

pip3 install SpeechRecognition
pip3 install -r ../requirements.txt

time python3 test_houndify.py
