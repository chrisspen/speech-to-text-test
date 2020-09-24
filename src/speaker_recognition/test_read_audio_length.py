#!/usr/bin/env python
import os

from pydub.utils import mediainfo

original_size = 0
split_size = 0

DIR = 'data2'
for fn in sorted(os.listdir(DIR)):
    fn = os.path.join(DIR, fn)
    print(fn)
    info = mediainfo(fn)
    if not info:
        continue
    duration = float(mediainfo(fn)['duration'])
    print(fn, duration)
    if 'silence' in fn:
        split_size += duration
    else:
        original_size += duration

print('split size:', split_size)
print('original size:', original_size)
