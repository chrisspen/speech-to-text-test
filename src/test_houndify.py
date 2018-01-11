#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-10
Evaluates the accuracy of Houndify.

Requires you enter your custom client ID and client key.

To get these:
1. register an account on houndify.com
2. create an "application"
3. select domain "speech-to-text only" and click "save and continue"

This should land you on a page showing your credentials.
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import sys

import speech_recognition as sr

from tester import BaseTester, RATE16K_MONO_WAV

from test_houndify_sensitive import *

class Tester(BaseTester):

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.recognizer = sr.Recognizer()

    def audio_to_text(self, fn):
        with sr.AudioFile(fn) as source:
            try:
                audio = self.recognizer.record(source)
                print('Running inference.', file=sys.stderr)
                inference_start = timer()
                text = self.recognizer.recognize_houndify(
                    audio,
                    client_id=HOUNDIFY_CLIENT_ID, # pylint: disable=undefined-variable
                    client_key=HOUNDIFY_CLIENT_KEY) # pylint: disable=undefined-variable
                print('text:', text)
                inference_end = timer() - inference_start
                print('Inference took %0.3fs audio file.' % (inference_end,), file=sys.stderr)
                return text
            except sr.UnknownValueError:
                print("Houndify could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Houndify service; {0}".format(e))
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
