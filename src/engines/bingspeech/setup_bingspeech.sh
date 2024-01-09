#!/bin/bash
# CS 2017.1.7
# Sets up the program DeepSpeech.
# Based on instructions at:
# https://github.com/mozilla/DeepSpeech

[ -d ../.env_bingspeech ] && rm -Rf ../.env_bingspeech
virtualenv -p python3 ../.env_bingspeech
. ../.env_bingspeech/bin/activate

pip install --only-binary scipy -r ../requirements.txt

time python test_bingspeech.py
