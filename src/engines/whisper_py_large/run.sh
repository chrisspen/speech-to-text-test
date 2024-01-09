#!/bin/bash
# CS 2024.1.6
# https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-en
set -e

source ../../common.sh

PYTHONVER=3.11
PIPVER=3
VENV=whisper_py

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML
pip$PIPVER install git+https://github.com/openai/whisper.git -q

time python test.py
