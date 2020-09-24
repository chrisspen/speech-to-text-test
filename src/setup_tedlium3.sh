#!/bin/bash
# CS 2019.11.21
set -e

[ -d ../.env_tedlium3 ] && rm -Rf ../.env_tedlium3
virtualenv -p python3.7 ../.env_tedlium3
. ../.env_tedlium3/bin/activate

pip install PyYAML

time python test_tedlium.py
