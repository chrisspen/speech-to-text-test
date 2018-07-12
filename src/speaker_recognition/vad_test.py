#!/usr/bin/env python
"""
Tests:

https://pypi.python.org/pypi/webrtcvad/
https://github.com/wangshub/python-vad/blob/master/vad.py
"""
import wave
import sys

import webrtcvad
import pyaudio
 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# A frame must be either 10, 20, or 30 ms in duration.
CHUNK_DURATION_MS = 30 # supports 10, 20 and 30 (ms)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # chunk to read
# CHUNK_BYTES = CHUNK_SIZE * 2 # 16bit = 2 bytes, PCM

vad = webrtcvad.Vad()
audio = pyaudio.PyAudio()
 
# start Recording
# The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, or 32000 Hz.
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE)
print "recording..."
count = 0
while 1:
    chunk = stream.read(CHUNK_SIZE)
    # print('chunk:', type(chunk))
    active = vad.is_speech(chunk, RATE)
    sys.stdout.write('\ractive: %s, count: %i' % (active, count))
    sys.stdout.flush()
    count += 1        
print "\nfinished recording"
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
