#!/bin/bash
# CS 2023.7.26
set -e

PYTHONVER=3

[ -d ../.env_whisper ]  && rm -Rf ../.env_whisper
python3 -m venv ../.env_whisper
. ../.env_whisper/bin/activate

pip$PYTHONVER install PyYAML

time python test_whisper.py
