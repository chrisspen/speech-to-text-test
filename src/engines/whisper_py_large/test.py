#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2023-7-26
Evaluates the accuracy of Whisper.

https://github.com/ggerganov/whisper.cpp
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import os
import sys
import re
import tempfile
import wave
from subprocess import getstatusoutput

import whisper

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'Whisper'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.model = whisper.load_model("large")

    def audio_to_text(self, fn):
        print('Running inference.', file=sys.stderr)
        fqfn = os.path.abspath(fn)
        inference_start = timer()

        result = self.model.transcribe(fqfn)
        # Returns a result of the form:
        # {'text': ' does that look good', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 3.92, 'text': ' does that look good', 'tokens': [50365, 775, 300, 574, 665, 50561], 'temperature': 0.0, 'avg_logprob': -0.28177268164498465, 'compression_ratio': 0.7037037037037037, 'no_speech_prob': 0.0403783954679966}], 'language': 'en'}

        inference_end = timer() - inference_start

        text = (result.get('text', '') or '').strip()
        print('raw response:', result)
        print('raw text:', repr(text))
        text = text.replace('[BLANK_AUDIO]', '').strip()
        text = text.replace('"', '') # It likes to try and insert quotes.
        text = re.sub(r'\([^\)]+\)', '', text, flags=re.MULTILINE).strip() # It likes to note sounds effects like "(mouse clicks)"
        text = re.sub(r'\[[^\]]+\]', '', text, flags=re.MULTILINE).strip() # It likes to note sounds effects like "[ silence ]"
        text = re.sub(r'[\s\t\n]+', ' ', text).strip() # compact whitespace
        print('cleaned text:', text)
        print('Inference took %0.3fs.' % (inference_end), file=sys.stderr)
        return text

if __name__ == '__main__':
    tester = Tester()
    tester.test()
