#!/bin/bash
# CS 2019.11.21
# Sets up the Word2Vec program from Facebook.
# Based on instructions at:
#
#   https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
#   https://becominghuman.ai/unsupervised-pre-training-for-speech-recognition-wav2vec-aba643824324
#
set -e

[ -d ../.env_word2vec ] && rm -Rf ../.env_word2vec
virtualenv -p python3.7 ../.env_word2vec
. ../.env_word2vec/bin/activate

mkdir -p ../data/models/word2vec
wget --output-document=../data/models/word2vec/wav2vec_large.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt

pip install fairseq librosa
pip install -r ../requirements.txt
