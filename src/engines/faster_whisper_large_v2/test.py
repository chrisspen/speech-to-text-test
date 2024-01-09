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

from faster_whisper import WhisperModel

sys.path.insert(0, '../..')

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'faster_whisper_large_v2'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.model = WhisperModel("large-v2", device="cpu", compute_type="int8")

    def audio_to_text(self, fn):
        print('Running inference.', file=sys.stderr)
        fqfn = os.path.abspath(fn)
        inference_start = timer()

        segments, info = self.model.transcribe(fqfn)
        # Returns a result of the form:
        # segments = [{start: end: text:}]

        inference_end = timer() - inference_start

        text = ' '.join(segment.text for segment in segments)
        text = text.strip()
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
