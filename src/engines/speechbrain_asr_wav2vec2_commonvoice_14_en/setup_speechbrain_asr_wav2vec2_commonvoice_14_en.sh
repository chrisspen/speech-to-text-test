#!/bin/bash
# CS 2024.1.6
# https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-en
set -e

source ./common.sh

PYTHONVER=3.11
PIPVER=3
VENV=speechbrain_asr_wav2vec2_commonvoice_14_en

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML

CWD=$PWD

cd /tmp
git clone https://github.com/speechbrain/speechbrain
cd speechbrain
git checkout unstable-v0.6
pip$PIPVER install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install -e .

cd $CWD

time python test_speechbrain_asr_wav2vec2_commonvoice_14_en.py
