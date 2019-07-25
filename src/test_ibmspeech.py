#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-10
Evaluates the accuracy of IBM Speech.
https://raw.githubusercontent.com/Uberi/speech_recognition/master/examples/audio_transcribe.py
"""
from __future__ import absolute_import, division, print_function
import time
from timeit import default_timer as timer

import sys

import speech_recognition as sr

from tester import BaseTester, RATE16K_MONO_WAV

from test_ibmspeech_sensitive import *

class Tester(BaseTester):

    name = 'IBMSpeech'

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
                text = self.recognizer.recognize_ibm(audio, key=KEY) # pylint: disable=undefined-variable

                # IBM adds some unnecessary punctuation, which our corpus doesn't check, so strip it out.
                text = text.replace('?', '')
                if text.endswith('.'):
                    text = text[:-1].strip()

                print('text:', text)
                inference_end = timer() - inference_start
                print('Inference took %0.3fs audio file.' % (inference_end,), file=sys.stderr)
                return text
            except sr.UnknownValueError:
                print("IBM Speech to Text-d5 could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from IBM Speech to Text-d5 service; {0}".format(e))
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
