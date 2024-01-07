#!/bin/bash
# CS 2024.1.6
# https://huggingface.co/speechbrain/asr-wav2vec2-librispeech
set -e

source ./common.sh

PYTHONVER=3.11
PIPVER=3
VENV=speechbrain

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML speechbrain transformers

time python test_speechbrain_asr_wav2vec2_librispeech.py
