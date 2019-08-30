#!/bin/bash
set -e

[ -d ../.env_amazon ]  && rm -Rf ../.env_amazon
virtualenv -p python3.7 ../.env_amazon
. ../.env_amazon/bin/activate

pip install --only-binary scipy -r ../requirements.txt
rm -Rf ../.env_amazon/src/speech-recognition/speech_recognition
ln -s /home/chris/git/speech_recognition/speech_recognition ../.env_amazon/src/speech-recognition/

time python test_amazon.py
