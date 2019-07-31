#!/bin/bash
set -e

PYTHONVER=3

[ -d ../.env_ibmspeech ]  && rm -Rf ../.env_ibmspeech
virtualenv -p python$PYTHONVER ../.env_ibmspeech
. ../.env_ibmspeech/bin/activate

pip$PYTHONVER install --only-binary scipy -r ../requirements.txt

time python$PYTHONVER test_ibmspeech.py
