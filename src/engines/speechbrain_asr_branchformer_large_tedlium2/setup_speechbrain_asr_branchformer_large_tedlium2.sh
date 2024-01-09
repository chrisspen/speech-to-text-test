#!/bin/bash
# CS 2024.1.6
# https://huggingface.co/speechbrain/asr-branchformer-large-tedlium2
set -e

source ./common.sh

PYTHONVER=3.11
PIPVER=3
VENV=speechbrain

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML speechbrain transformers

time python test_speechbrain_asr_branchformer_large_tedlium2.py
