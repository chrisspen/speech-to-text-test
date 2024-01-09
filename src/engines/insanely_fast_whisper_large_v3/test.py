#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS 2023-7-26
Evaluates the accuracy of Whisper.

https://github.com/ggerganov/whisper.cpp
"""
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import os
import sys
import re
import tempfile
import wave
from subprocess import getstatusoutput

# from faster_whisper import WhisperModel
from transformers import pipeline
import torch

sys.path.insert(0, '../..')

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'insanely_fast_whisper_large_v3'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        # self.model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        device_id = "cpu"
        # dtype = torch.float16
        dtype = torch.float32 if device_id == "cpu" else torch.float16
        model_name = 'openai/whisper-large-v3'
        flash = False
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=dtype,
            device=device_id,
            model_kwargs={"attn_implementation": "flash_attention_2"} if flash else {"attn_implementation": "sdpa"},
        )

    def audio_to_text(self, fn):
        print('Running inference.', file=sys.stderr)
        fqfn = os.path.abspath(fn)
        inference_start = timer()

        timestamp = 'chunk'
        ts = "word" if timestamp == "word" else True
        generate_kwargs = {"task": 'transcribe', "language": None}
        outputs = self.pipe(
            fqfn,
            chunk_length_s=30,
            batch_size=24,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
        )

        inference_end = timer() - inference_start
        print('raw output:', outputs)
        text = outputs["text"]
        text = text.strip()
        print('raw text:', repr(text))
        text = text.replace('[BLANK_AUDIO]', '').strip()
        text = text.replace('"', '') # It likes to try and insert quotes.
        text = re.sub(r'\([^\)]+\)', '', text, flags=re.MULTILINE).strip() # It likes to note sounds effects like "(mouse clicks)"
        text = re.sub(r'\[[^\]]+\]', '', text, flags=re.MULTILINE).strip() # It likes to note sounds effects like "[ silence ]"
        text = re.sub(r'[\s\t\n]+', ' ', text).strip() # compact whitespace
        print('cleaned text:', text)
        print('Inference took %0.3fs.' % (inference_end), file=sys.stderr)
        return text

if __name__ == '__main__':
    tester = Tester()
    tester.test()
