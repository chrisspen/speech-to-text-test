#!/usr/bin/env python
import os
import re
from math import exp
from time import time
from subprocess import getoutput

import json

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'Tedlium3'

    audio_format = RATE16K_MONO_WAV

    def audio_to_text(self, fn):
        fn = os.path.abspath(fn)
        print('Processing %s.' % fn)
        time_start = time()
        output = getoutput(f'cd /home/chris/git/espnet/egs/tedlium3/asr1; ../../../utils/recog_wav.sh --models tedlium3.transformer.v1 {fn}')
        td = time() - time_start
        print('output:', output)

        name = os.path.splitext(os.path.split(fn)[-1])[0]
        print('name:', name)
        result_dir = os.path.join('/home/chris/git/espnet/egs/tedlium3/asr1/decode', name)
        result_fn = os.path.join(result_dir, 'result.json')
        result = json.load(open(result_fn))
        print('json:', result)
        text = result['utts'][name]['output'][0]['rec_text']
        raw_score = result['utts'][name]['output'][0]['score']
        print('raw_score:', raw_score)
        certainty = 1 - exp(raw_score)
        print('certainty:', certainty)

        print('raw_text:', text)
        delimiter = '▁'
        text = text.replace(delimiter, ' ').replace('-', ' ')
        text = re.sub(r'<[^>]+>', ' ', text) # Remove all "<code>" elements.
        text = re.sub(r'[ ]+', ' ', text)
        text = text.strip()
        print('text:', text)
        # print('certainty:', certainty)
        print("%s decoding took %8.2fs, certainty: %f" % (fn, td, certainty))
        # print(text, certainty)
        return text


if __name__ == '__main__':
    tester = Tester()
    tester.test()
