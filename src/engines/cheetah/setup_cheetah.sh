#!/bin/bash
# CS 2017.1.7
# Sets up the program Picovoice Cheetah.
#
set -e

[ -d ../.env_cheetah ] && rm -Rf ../.env_cheetah
virtualenv -p python3.7 ../.env_cheetah
. ../.env_cheetah/bin/activate

pip install -r ../requirements.txt

time python test_cheetah.py

