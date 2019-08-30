#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-10
Evaluates the accuracy of Amazon STT.
https://aws.amazon.com/transcribe/pricing/?nc=sn&loc=3
"""
from __future__ import absolute_import, division, print_function
import time
from timeit import default_timer as timer

import sys

import speech_recognition as sr

from tester import BaseTester, RATE16K_MONO_WAV

from test_amazon_sensitive import *

class Tester(BaseTester):

    name = 'Amazon'

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
                text = self.recognizer.recognize_amazon(
                    audio,
                    bucket_name='7dd8e643287e4179ab316d414dfefcfa',
                    access_key_id=ACCESS_KEY_ID,
                    secret_access_key=SECRET_ACCESS_KEY,
                    region='us-east-1')

                text = text.replace('?', '')
                if text.endswith('.'):
                    text = text[:-1].strip()

                print('text:', text)
                inference_end = timer() - inference_start
                print('Inference took %0.3fs audio file.' % (inference_end,), file=sys.stderr)
                return text
            except sr.UnknownValueError:
                print("Amazon could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Amazon service; {0}".format(e))
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
