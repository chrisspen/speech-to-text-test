#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-10
Evaluates the accuracy of PocketSphinx.
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import sys

import speech_recognition as sr

from tester import BaseTester, RATE16K_MONO_WAV

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
                text = self.recognizer.recognize_sphinx(audio)
                print('text:', text)
                inference_end = timer() - inference_start
                print('Inference took %0.3fs audio file.' % (inference_end,), file=sys.stderr)
                return text
            except sr.UnknownValueError:
                print("Sphinx could not understand audio")
            except sr.RequestError as e:
                print("Sphinx error; {0}".format(e))
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
