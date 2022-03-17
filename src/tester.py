from __future__ import absolute_import, division, print_function

import os
from difflib import SequenceMatcher

import yaml

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

ANNOTATIONS_FN = 'annotations.yaml'
RATE16K_MONO_WAV = 'rate16k-mono.wav'
AUDIO_DIR = '../data/audio'


def rev_accuracy(r, h):
    # Inverse of the approximate word error rate, fastest and most like Rev's metric.
    # print('rev_accuracy.r:', r)
    # print('rev_accuracy.h:', h)
    # https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher.ratio
    r = r.strip().lower().replace("'", "").replace(".", "").replace(",", "")
    h = h.strip().lower().replace("'", "").replace(".", "").replace(",", "")
    r = r.split()
    h = h.split()
    # return 1 - SequenceMatcher(None, r, h).ratio() # error rate
    acc = SequenceMatcher(None, r, h).ratio() # accuracy rate
    # print('rev_accuracy.acc:', acc)
    return acc


class BaseTester:

    name = None

    # The audio encoding format required for input.
    audio_format = None

    def audio_to_text(self, fn):
        raise NotImplementedError

    def test(self):
        assert self.name
        assert self.audio_format, 'No audio format specified.'
        history = []
        rev_history = []
        abs_history = []
        data = yaml.safe_load(open(os.path.join(AUDIO_DIR, ANNOTATIONS_FN))) or {}
        i = 0
        total = len(data)
        for fn in sorted(data):
            i += 1
            print('Processing %i of %i...' % (i, total))
            if self.audio_format and self.audio_format not in fn:
                print('Skipping audio format that does not match %s.' % self.audio_format)
                continue
            print(fn)
            predicted_text = self.audio_to_text(os.path.join(AUDIO_DIR, fn))
            expected_text = data[fn]
            predicted_text = predicted_text.strip().lower().replace("'", "")
            expected_text = expected_text.strip().lower().replace("'", "")
            ratio = similar(predicted_text, expected_text)
            rev_acc = rev_accuracy(r=expected_text, h=predicted_text)
            print('\tpredicted_text:', predicted_text)
            print('\texpected_text: ', expected_text)
            print('\tmatch:', ratio, rev_acc)
            history.append(ratio)
            rev_history.append(rev_acc)
            abs_history.append(int(expected_text == predicted_text))

        print('='*80)
        accuracy = sum(history)/len(history)
        avg_rev_accuracy = sum(rev_history)/len(rev_history)
        abs_accuracy = sum(abs_history)/float(len(abs_history))
        print('diff accuracy:', accuracy)
        print('abs accuracy:', abs_accuracy)
        print('avg rev accuracy:', avg_rev_accuracy)
