#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2024-1-6

https://pytorch.org/hub/snakers4_silero-models_stt/
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import os
import sys
import traceback
from glob import glob

import torch

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'speechbrain_asr_wav2vec2_librispeech'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

        self.device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

        self.model, self.decoder, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='en', # also available 'de', 'es'
                                               device=self.device)
        self.read_batch, self.split_into_batches, self.read_audio, self.prepare_model_input = self.utils


    def audio_to_text(self, fn):
        print('Running inference.', file=sys.stderr)
        fqfn = os.path.abspath(fn)
        inference_start = timer()

        text = ''
        try:
            test_files = glob(fqfn)
            batches = self.split_into_batches(test_files, batch_size=10)
            model_input = self.prepare_model_input(self.read_batch(batches[0]), device=self.device)
            output = self.model(model_input)
            print('len:', len(output))
            assert len(output) == 1
            for example in output:
                text = self.decoder(example.cpu())
                break
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
