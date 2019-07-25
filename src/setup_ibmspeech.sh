#!/bin/bash

#TODO: enable when upstream supports IBM apikey
#[ ! -d ../.env_ibmspeech ] && virtualenv ../.env_ibmspeech
. ../.env_ibmspeech/bin/activate

pip install SpeechRecognition
pip install --only-binary scipy -r ../requirements.txt

python test_ibmspeech.py
