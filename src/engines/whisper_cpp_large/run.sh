#!/bin/bash
# CS 2023.7.26
# https://github.com/ggerganov/whisper.cpp
set -e

source ../../common.sh

PYTHONVER=3.11
PIPVER=3
VENV=whisper_cpp_large

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML
pip$PIPVER install git+https://github.com/aarnphm/whispercpp.git -vv

#time python test.py
