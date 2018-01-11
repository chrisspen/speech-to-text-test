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

class Tester(BaseTester):

    name = 'Ensemble'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

        # Absolute accuracy * differential accuracy used as weight.
        self.weights = {
            GoogleSpeechTester: 0.6 + 0.726952838341,
            PocketSphinxTester: 0.3 + 0.650675023999,
            DeepSpeechTester: 0.43 + 0.728972718858,
            HoundifyTester: 0.6 + 0.853484911794
        }
        self.testers = [
            GoogleSpeechTester(),
            PocketSphinxTester(),
            DeepSpeechTester(),
            HoundifyTester()
        ]

    def audio_to_text(self, fn):
        votes = defaultdict(float) # {text: sum of weights}
        for _tester in self.testers:
            predicted_text = _tester.audio_to_text(fn)
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
