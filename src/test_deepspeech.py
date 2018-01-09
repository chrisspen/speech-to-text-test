#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-8
Evaluates the accuracy of DeepSpeech.
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import os
import argparse
import sys
from difflib import SequenceMatcher

import scipy.io.wavfile as wav

from deepspeech.model import Model

import yaml

def similar(a, b):
	return SequenceMatcher(None, a, b).ratio()

# These constants control the beam search decoder

annotations_fn = 'annotations.yaml'
RATE16K_MONO = 'rate16k-mono'
AUDIO_DIR = '../data/audio'
DATA_DIR = '../data/models/deepspeech'

args_lm = os.path.join(DATA_DIR, 'models/lm.binary')
args_trie = os.path.join(DATA_DIR, 'models/trie')
args_model = os.path.join(DATA_DIR, 'models/output_graph.pb')
args_alphabet = os.path.join(DATA_DIR, 'models/alphabet.txt')
files = [args_lm, args_trie, args_model, args_alphabet]
for f in files:
	assert os.path.isfile(f)

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

print('Loading model from file %s' % (args_model), file=sys.stderr)
model_load_start = timer()
ds = Model(args_model, N_FEATURES, N_CONTEXT, args_alphabet, BEAM_WIDTH)
model_load_end = timer() - model_load_start
print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)

if args_lm and args_trie:
	print('Loading language model from files %s %s' % (args_lm, args_trie), file=sys.stderr)
	lm_load_start = timer()
	ds.enableDecoderWithLM(args_alphabet, args_lm, args_trie, LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
	lm_load_end = timer() - lm_load_start
	print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)

def tts(fn):
	#args_audio = os.path.join(AUDIO_DIR, 'guess-what-1.rate16k-mono.wav')
	fs, audio = wav.read(fn)
	# We can assume 16kHz
	audio_length = len(audio) * ( 1 / 16000)
	assert fs == 16000, "Only 16000Hz input WAV files are supported for now!"

	print('Running inference.', file=sys.stderr)
	inference_start = timer()
	text = ds.stt(audio, fs)
	print('text:', text)
	inference_end = timer() - inference_start
	print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
	return text

history = []
data = yaml.load(open(os.path.join(AUDIO_DIR, annotations_fn))) or {}
i = 0
total = len(data)
for fn in sorted(data):
	i += 1
	print('Processing %i of %i...' % (i, total))
	if RATE16K_MONO not in fn:
		continue
	print(fn)
	predicted_text = tts(os.path.join(AUDIO_DIR, fn))
	expected_text = data[fn]
	predicted_text = predicted_text.strip().lower().replace("'", "")
	expected_text = expected_text.strip().lower().replace("'", "")
	ratio = similar(predicted_text, expected_text)
	print('\tpredicted_text:', predicted_text)
	print('\texpected_text:', expected_text)
	print('\tmatch:', ratio)
	history.append(ratio)

print('='*80)
accuracy = sum(history)/len(history)
print('accuracy:', accuracy)
