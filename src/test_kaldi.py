#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2019-11-21
Evaluates the accuracy of Kaldi.
https://github.com/gooofy/zamia-speech#get-started-with-our-pre-trained-models
"""
from __future__ import absolute_import, division, print_function

import logging
from time import time

from kaldiasr.nnet3 import KaldiNNet3OnlineModel, KaldiNNet3OnlineDecoder

from tester import BaseTester, RATE16K_MONO_WAV

DEFAULT_MODELDIR    = '/opt/kaldi/model/kaldi-generic-en-tdnn_f'

class Options:
    pass

options = Options()
options.modeldir = DEFAULT_MODELDIR

logging.basicConfig(level=logging.DEBUG)

logging.debug('%s loading model...', options.modeldir)
time_start = time()
kaldi_model = KaldiNNet3OnlineModel(options.modeldir, acoustic_scale=1.0, beam=7.0, frame_subsampling_factor=3)
logging.debug('%s loading model... done, took %fs.', options.modeldir, time()-time_start)

logging.debug('%s creating decoder...', options.modeldir)
time_start = time()
decoder = KaldiNNet3OnlineDecoder (kaldi_model)
logging.debug('%s creating decoder... done, took %fs.', options.modeldir, time()-time_start)


class Tester(BaseTester):

    name = 'Kaldi'

    audio_format = RATE16K_MONO_WAV

    def audio_to_text(self, fn):
        _time_start = time()
        if decoder.decode_wav_file(fn):
            s, l = decoder.get_decoded_string()
            print('text:', s)
            td = time() - _time_start
            logging.debug("%s decoding took %8.2fs, likelyhood: %f", fn, td, l)
            return s
        logging.error("decoding of %s failed.", fn)
        return ''

if __name__ == '__main__':
    tester = Tester()
    tester.test()
