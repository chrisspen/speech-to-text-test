#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/Uberi/speech_recognition

[ ! -d ../.env_houndify ] && virtualenv ../.env_houndify
. ../.env_houndify/bin/activate

pip install SpeechRecognition
pip install -r ../requirements.txt

python test_houndify.py
