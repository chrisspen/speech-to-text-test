#!/usr/bin/env python
"""
Splits a sample Mp3 into multiple audio files based on periods of silence.
"""

from pydub import AudioSegment
from pydub.silence import split_on_silence

fn = "data/sample1.mp3"
print('Reading file:', fn)
sound_file = AudioSegment.from_mp3(fn)
print('Calculating splits on silence...')
audio_chunks = split_on_silence(
    sound_file,
    # must be silent for at least half a second
    min_silence_len=500,
    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16,
    keep_silence=True
)

print('Iterating...')
for i, chunk in enumerate(audio_chunks):
    out_file = "data/sample1-chunk{0}.mp3".format(i)
    print("Saving:", out_file)
    chunk.export(out_file, format="mp3")
