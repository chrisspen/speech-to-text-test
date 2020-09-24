#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2018-1-10
Evaluates the accuracy of PocketSphinx.
"""
from __future__ import absolute_import, division, print_function

from collections import defaultdict

from tester import BaseTester, RATE16K_MONO_WAV
from test_googlespeech import Tester as GoogleSpeechTester
from test_pocketsphinx import Tester as PocketSphinxTester
from test_deepspeech import Tester as DeepSpeechTester
from test_houndify import Tester as HoundifyTester
# from test_bingspeech import Tester as BingTester
from test_ibmspeech import Tester as IBMTester
from test_kaldi import Tester as KaldiTester
from test_jasper import Tester as JasperTester

class Tester(BaseTester):

    name = 'Ensemble'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

        # Absolute accuracy * differential accuracy used as weight.
        self.weights = {
            GoogleSpeechTester: (0.34 + 0.81)/2.,
            PocketSphinxTester: (0.17 + 0.66)/2.,
            DeepSpeechTester: (0.32 + 0.76)/2.,
            HoundifyTester: (0.38 + 0.89)/2.,
            IBMTester: (0.32 + 0.80)/2.,
            KaldiTester: (0.42 + 0.85)/2.,
            JasperTester: (0.32 + 0.85)/2.,
            # BingTester: (0.666666666667 + 0.837140933267)/2.,
        }
        self.testers = [
            GoogleSpeechTester(),
            # PocketSphinxTester(),
            # DeepSpeechTester(),
            # HoundifyTester(),
            # IBMTester(),
            KaldiTester(),
            JasperTester(),
            # BingTester(delay=False),
        ]

    def audio_to_text(self, fn):
        votes = defaultdict(float) # {text: sum of weights}
        for _tester in self.testers:
            predicted_text = _tester.audio_to_text(fn) or ''
            predicted_text = predicted_text.lower().strip()
            predicted_text = predicted_text.replace("'", '')
            weight = self.weights[type(_tester)]
            if predicted_text:
                # If most of the testers give up, don't let their failures result in a failed prediction.
                # Only look at positive responses.
                votes[predicted_text] += weight
                print('\tvote:', type(_tester).name, '=>', predicted_text)
        print('votes:', votes)
        if votes:
            best_text, best_weight = sorted(votes.items(), key=lambda o: o[1])[-1]
            print('best:', best_text, best_weight)
            return best_text
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
