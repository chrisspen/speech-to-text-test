#!/bin/bash
set -e

PYTHONVER=3

[ -d ../.env_assemblyai ]  && rm -Rf ../.env_assemblyai
virtualenv -p python$PYTHONVER ../.env_assemblyai
. ../.env_assemblyai/bin/activate

pip$PYTHONVER install assemblyai PyYAML

time python$PYTHONVER test_assemblyai.py
