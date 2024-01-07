#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2024-1-6

https://huggingface.co/speechbrain/asr-wav2vec2-librispeech/blob/main/README.md
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import os
import sys
import re
import tempfile
import wave
import traceback
from subprocess import getstatusoutput

from speechbrain.pretrained import EncoderASR

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'speechbrain_asr_wav2vec2_librispeech'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")

    def audio_to_text(self, fn):
        print('Running inference.', file=sys.stderr)
        fqfn = os.path.abspath(fn)
        inference_start = timer()

        text = ''
        try:
            text = self.asr_model.transcribe_file(fqfn)
        except RuntimeError as exc:
            print('Error transcribing file %s: %s' % (fqfn, exc))
            traceback.print_exc()

        inference_end = timer() - inference_start

        print('raw text:', repr(text))
        text = (text or '').strip().lower()
        print('cleaned text:', text)
        print('Inference took %0.3fs.' % (inference_end), file=sys.stderr)
        return text

if __name__ == '__main__':
    tester = Tester()
    tester.test()