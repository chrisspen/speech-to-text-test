#!/bin/bash
set -e

PYTHONVER=3

[ -d ../.env_vosk ]  && rm -Rf ../.env_vosk
virtualenv -p python$PYTHONVER ../.env_vosk
. ../.env_vosk/bin/activate

pip$PYTHONVER install -r requirements-vosk.txt

time python$PYTHONVER src/test_vosk_lgraph.py
time python$PYTHONVER src/test_vosk_big.py
