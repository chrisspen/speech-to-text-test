#!/bin/bash
# CS 2019.11.21
set -e

[ -d ../.env_paddle ] && rm -Rf ../.env_paddle
virtualenv -p python3.7 ../.env_paddle
. ../.env_paddle/bin/activate

pip install PyYAML

time python test_paddle.py
