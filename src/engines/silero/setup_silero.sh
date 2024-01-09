#!/bin/bash
# CS 2024.1.6
# https://huggingface.co/speechbrain/asr-wav2vec2-librispeech
set -e

source ./common.sh

PYTHONVER=3.11
PIPVER=3
VENV=silero

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML torchaudio omegaconf soundfile

time python test_silero.py
