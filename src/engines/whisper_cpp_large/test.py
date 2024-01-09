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

from whispercpp import Whisper

sys.path.insert(0, '../..')

from tester import BaseTester, RATE16K_MONO_WAV

# Assumes Whisper.cpp is checked out and built in a parallel folder to our current repo.
# https://github.com/ggerganov/whisper.cpp
# WHISPER_BIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../whisper.cpp/main')
# assert os.path.isfile(WHISPER_BIN), 'Whisper missing: %s' % WHISPER_BIN

class Tester(BaseTester):

    name = 'whisper_cpp_large'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.w = Whisper.from_pretrained('large')

    def audio_to_text(self, fn):
        print('Running inference.', file=sys.stderr)
        fqfn = os.path.abspath(fn)
        inference_start = timer()

        result = self.w.transcribe(fqfn)
        text = self.w.extract_text(result)
        text = (text or [''])[0]

        inference_end = timer() - inference_start

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
