#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys
import wave
from functools import reduce

import json

from vosk import Model, KaldiRecognizer, SetLogLevel

from tester import BaseTester, RATE16K_MONO_WAV

SetLogLevel(0)

class Tester(BaseTester):

    name = 'Vosk-Big'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_fn = "../data/models/vosk-model-en-us-0.22"
        assert os.path.isdir(model_fn)
        self.model = Model(model_fn)

    def audio_to_text(self, fn):

        wf = wave.open(fn, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print ("Audio file must be WAV format mono PCM.")
            sys.exit(1)

        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        result = rec.FinalResult()
        result = json.loads(result)

        conf = reduce(lambda x,y: x*y, [r['conf'] for r in result['result']], 1)
        print('conf:', conf)
        return result['text']


if __name__ == '__main__':
    tester = Tester()
    tester.test()
