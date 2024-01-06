from __future__ import absolute_import, division, print_function

import os
import re
from difflib import SequenceMatcher

import yaml

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

ANNOTATIONS_FN = 'annotations.yaml'
RATE16K_MONO_WAV = 'rate16k-mono.wav'
AUDIO_DIR = '../data/audio'


def clean_text(text):
    text = text or ''
    text = text.replace('-', ' ').strip().lower().replace("'", "")
    text = re.sub(r'[\.\!\?\,]+', '', text)
    return text


def rev_accuracy(r, h):
    # Inverse of the approximate word error rate, fastest and most like Rev's metric.
    # https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher.ratio
    r = r.replace('-', ' ')
    h = h.replace('-', ' ')
    r = re.sub(r'[\.\!\?\,]+', '', r)
    h = re.sub(r'[\.\!\?\,]+', '', h)
    r = r.strip().lower().replace("'", "").replace(".", "").replace(",", "")
    h = h.strip().lower().replace("'", "").replace(".", "").replace(",", "")
    r = r.split()
    h = h.split()
    acc = SequenceMatcher(None, r, h).ratio() # accuracy rate
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
            print('self.audio_format:', self.audio_format, fn)
            if self.audio_format and self.audio_format not in fn:
                print('Skipping audio format that does not match %s.' % self.audio_format)
                continue
            print(fn)
            predicted_text = self.audio_to_text(os.path.join(AUDIO_DIR, fn))
            expected_text = data[fn]
            predicted_text = clean_text(predicted_text)
            expected_text = clean_text(expected_text)
            ratio = similar(predicted_text, expected_text)
            rev_acc = rev_accuracy(r=expected_text, h=predicted_text)
            abs_acc = int(expected_text == predicted_text)
            print('\tpredicted_text:', predicted_text)
            print('\texpected_text: ', expected_text)
            print('\tSequence Ratio:', ratio)
            print('\tAbs Acc:', abs_acc)
            print('\tRev Acc:', rev_acc)
            history.append(ratio)
            rev_history.append(rev_acc)
            abs_history.append(abs_acc)

        print('='*80)
        accuracy = sum(history)/len(history)
        avg_rev_accuracy = sum(rev_history)/len(rev_history)
        abs_accuracy = sum(abs_history)/float(len(abs_history))
        print('diff accuracy:', accuracy)
        print('abs accuracy:', abs_accuracy)
        print('avg rev accuracy:', avg_rev_accuracy)
