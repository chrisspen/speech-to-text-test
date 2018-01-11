from __future__ import absolute_import, division, print_function

import os
from difflib import SequenceMatcher

import yaml

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

ANNOTATIONS_FN = 'annotations.yaml'
RATE16K_MONO_WAV = 'rate16k-mono.wav'
AUDIO_DIR = '../data/audio'

class BaseTester(object):

    # The audio encoding format required for input.
    audio_format = None

    def audio_to_text(self, fn):
        raise NotImplementedError

    def test(self):
        assert self.audio_format, 'No audio format specified.'
        history = []
        abs_history = []
        data = yaml.load(open(os.path.join(AUDIO_DIR, ANNOTATIONS_FN))) or {}
        i = 0
        total = len(data)
        for fn in sorted(data):
            i += 1
            print('Processing %i of %i...' % (i, total))
            if self.audio_format and self.audio_format not in fn:
                continue
            print(fn)
            predicted_text = self.audio_to_text(os.path.join(AUDIO_DIR, fn))
            expected_text = data[fn]
            predicted_text = predicted_text.strip().lower().replace("'", "")
            expected_text = expected_text.strip().lower().replace("'", "")
            ratio = similar(predicted_text, expected_text)
            print('\tpredicted_text:', predicted_text)
            print('\texpected_text:', expected_text)
            print('\tmatch:', ratio)
            history.append(ratio)
            abs_history.append(int(expected_text == predicted_text))

        print('='*80)
        accuracy = sum(history)/len(history)
        abs_accuracy = sum(abs_history)/float(len(abs_history))
        print('diff accuracy:', accuracy)
        print('abs accuracy:', abs_accuracy)
