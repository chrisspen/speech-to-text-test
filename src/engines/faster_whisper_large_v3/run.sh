#!/bin/bash
# CS 2024.1.8
# https://github.com/SYSTRAN/faster-whisper
set -e

source ../../common.sh

PYTHONVER=3.11
PIPVER=3
VENV=faster_whisper_large_v3

init_venv $VENV $PYTHONVER $PIPVER

activate_venv $VENV

pip$PIPVER install PyYAML faster-whisper

time python test.py
