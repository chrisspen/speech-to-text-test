#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2019-11-21
Evaluates the accuracy of Kaldi.
https://github.com/gooofy/zamia-speech#get-started-with-our-pre-trained-models
"""
from __future__ import absolute_import, division, print_function

import tempfile
import os
import logging
import re
from math import exp
from time import time

from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader

from tester import BaseTester, RATE16K_MONO_WAV

def likelyhood_to_certainty(v):
    try:
        odds = exp(v)
        certainty = odds / (1. + odds)
    except OverflowError:
        certainty = 1.0
    return certainty


class Tester(BaseTester):

    name = 'Aspire'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        print('Initializing aspire model...')
        decoder_opts = LatticeFasterDecoderOptions()
        decoder_opts.beam = 13
        decoder_opts.max_active = 7000
        decodable_opts = NnetSimpleComputationOptions()
        decodable_opts.acoustic_scale = 1.0
        decodable_opts.frame_subsampling_factor = 3
        decodable_opts.frames_per_chunk = 150
        self.asr = NnetLatticeFasterRecognizer.from_files(
            "/home/chris/git/pykaldi/examples/setups/aspire/exp/tdnn_7b_chain_online/final.mdl",
            "/home/chris/git/pykaldi/examples/setups/aspire/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst",
            "/home/chris/git/pykaldi/examples/setups/aspire/data/lang/words.txt",
            decoder_opts=decoder_opts,
            decodable_opts=decodable_opts)

        _, fn = tempfile.mkstemp()
        os.remove(fn)
        self.scp_fn = scp_fn = '%s.scp' % fn

        # Define feature pipelines as Kaldi rspecifiers
        self.feats_rspec = (
            f"ark:compute-mfcc-feats --config=/home/chris/git/pykaldi/examples/setups/aspire/conf/mfcc_hires.conf scp:{scp_fn} ark:- |"
        )
        self.ivectors_rspec = (
            f"ark:compute-mfcc-feats --config=/home/chris/git/pykaldi/examples/setups/aspire/conf/mfcc_hires.conf scp:{scp_fn} ark:- |"
            f"ivector-extract-online2 --config=/home/chris/git/pykaldi/examples/setups/aspire/conf/ivector_extractor.conf " \
            f"ark:/home/chris/git/pykaldi/examples/setups/aspire/data/test/spk2utt ark:- ark:- |"
        )

    def audio_to_text(self, fn):
        fn = os.path.abspath(fn)
        print('Processing %s.' % fn)
        with open(self.scp_fn, 'w') as fout:
            fout.write('utt1 %s' % fn)
        with SequentialMatrixReader(self.feats_rspec) as f, SequentialMatrixReader(self.ivectors_rspec) as i:#, open("out/test/decode.out", "w") as o:
            for (key, feats), (_, ivectors) in zip(f, i):
                time_start = time()
                out = self.asr.decode((feats, ivectors))
                td = time() - time_start
                # print(key, out["text"], file=o)
                print('out:', out)
                likelihood = out['likelihood']
                certainty = likelyhood_to_certainty(likelihood)
                print('certainty:', certainty)
                logging.info("%s decoding took %8.2fs, certainty: %f", fn, td, certainty)
                text = out["text"]
                print('text:', text)
                text = re.sub(r'\[[^\]]+\]', '', text) # remove "[noise]"
                text = re.sub(r'[ ]+', ' ', text) # collapse whitespace
                print('text2:', text)
                return text


if __name__ == '__main__':
    tester = Tester()
    tester.test()
