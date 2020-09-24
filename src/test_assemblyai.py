#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import time
import sys

from timeit import default_timer as timer

import requests

from tester import BaseTester, RATE16K_MONO_WAV

from test_assemblyai_sensitive import *

class Tester(BaseTester):

    name = 'AssemblyAI'

    audio_format = RATE16K_MONO_WAV

    def audio_to_text(self, fn):
        print('Running inference on %s.' % fn, file=sys.stderr)
        inference_start = timer()

        filename = fn
        assert os.path.isfile(fn)

        def read_file(filename, chunk_size=5242880):
            with open(filename, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data

        # Upload file.
        headers = {'authorization': API_TOKEN}
        response = requests.post('https://api.assemblyai.com/v2/upload',
                                 headers=headers,
                                 data=read_file(filename))
        print(response.json())
        upload_url = response.json()['upload_url']
        print('upload_url:', upload_url)

        # Queue file for transcription.
        endpoint = "https://api.assemblyai.com/v2/transcript"
        json = {
          "audio_url": upload_url
        }
        headers = {
            "authorization": API_TOKEN,
            "content-type": "application/json"
        }
        response = requests.post(endpoint, json=json, headers=headers)
        data = response.json()
        print(data)
        transciption_id = data['id']

        # Wait for transcription.
        confidence = None
        text = ''
        while 1:
            endpoint = f"https://api.assemblyai.com/v2/transcript/{transciption_id}"
            headers = {
                "authorization": API_TOKEN,
            }
            response = requests.get(endpoint, headers=headers)
            data = response.json()
            print(data)
            status = data['status']
            if status in ('completed', 'error'):
                confidence = data['confidence']
                text = data['text']
                break
            print('Waiting...')
            time.sleep(5)

        # They add some unnecessary punctuation, which our corpus doesn't check, so strip it out.
        text = text.replace('?', '')
        if text.endswith('.'):
            text = text[:-1].strip()

        print('text:', text)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs audio file.' % (inference_end,), file=sys.stderr)
        return text

if __name__ == '__main__':
    tester = Tester()
    tester.test()
