#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-10
Evaluates the accuracy of BingSpeech.
https://raw.githubusercontent.com/Uberi/speech_recognition/master/examples/audio_transcribe.py
"""
from __future__ import absolute_import, division, print_function
import time
from timeit import default_timer as timer

import sys

import speech_recognition as sr

from tester import BaseTester, RATE16K_MONO_WAV

from test_bingspeech_sensitive import *

class Tester(BaseTester):

    name = 'BingSpeech'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        self.delay = kwargs.pop('delay', True)
        super(Tester, self).__init__(*args, **kwargs)
        self.recognizer = sr.Recognizer()

    def audio_to_text(self, fn):
        with sr.AudioFile(fn) as source:
            try:
                audio = self.recognizer.record(source)
                print('Running inference.', file=sys.stderr)
                inference_start = timer()
                if self.delay:
                    time.sleep(60/20.+0.5) # bing limits us to 20 requests per minute...ugh
                text = self.recognizer.recognize_bing(audio, key=BING_KEY) # pylint: disable=undefined-variable

                # Bing adds some unnecessary punctuation, which our corpus doesn't check, so strip it out.
                text = text.replace('?', '')
                if text.endswith('.'):
                    text = text[:-1].strip()

                print('text:', text)
                inference_end = timer() - inference_start
                print('Inference took %0.3fs audio file.' % (inference_end,), file=sys.stderr)
                return text
            except sr.UnknownValueError:
                print("Microsoft Bing Voice Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
