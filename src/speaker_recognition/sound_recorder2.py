#!/usr/bin/env python
from __future__ import print_function
import pyaudio
import struct
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


RATE = 16000 #44100
# RATE = 44100
INPUT_BLOCK_TIME = 0.03 # 30 ms
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME) # samples/second * seconds = samples

def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))

class AudioHandler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.plot_counter = 0

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        device_index = None
        for i in range( self.pa.get_device_count() ):
            devinfo = self.pa.get_device_info_by_index(i)
            print('Device %{}: %{}'.format(i, devinfo['name']))

            for keyword in ['mic','input']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device {} - {}'.format(i, devinfo['name']))
                    device_index = i
                    return device_index

        if device_index == None:
            print('No preferred input found; using default input device.')

        return device_index

    def open_mic_stream( self ):
        device_index = self.find_input_device()
        print('device_index:', device_index)

        stream = self.pa.open(  format = pyaudio.paInt16,
                                channels = 1,
                                rate = RATE,
                                input = True,
                                input_device_index = device_index,
                                frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

        return stream

    def processBlock(self, snd_block):
        print('sample.shape:', snd_block.shape)
        f, t, Sxx = signal.spectrogram(snd_block, RATE)
        # f, t, Sxx = signal.spectrogram(snd_block, RATE, nperseg=64)#bad for voice
        # f, t, Sxx = signal.spectrogram(snd_block, RATE, nperseg=64, nfft=256, noverlap=60)
        # f, t, Sxx = signal.spectrogram(snd_block, RATE, noverlap=250)
        
        # Limit frequencies to the human voice range.
        # https://stackoverflow.com/a/48106480/247542
        
        fmin = 50 # Hz
        fmax = 300 # Hz
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice,:][0]
        Sxx = Sxx.flatten()
        print('Sxx.shape:', Sxx.shape)
        print('Sxx:', Sxx)
        
        # print('Sxx:', Sxx)
        # dBS = 10 * np.log10(Sxx)  # convert to dB
        # # print('dBS:', dBS)
        # # plt.pcolormesh(t, f, Sxx)
        # plt.pcolormesh(t, f, dBS)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.savefig('data/spec{}.png'.format(self.plot_counter), bbox_inches='tight')
        # self.plot_counter += 1

    def listen(self):
        try:
            raw_block = self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow = False)
            count = len(raw_block) / 2
            format = '%dh' % (count)
            snd_block = np.array(struct.unpack(format, raw_block))
        except Exception as e:
            print('Error recording: {}'.format(e))
            return

        amplitude = get_rms(snd_block)
        # print('amplitude:', amplitude)
        from math import log10
        # dB = 20 * log10(amplitude)
        # print('dB:', dB)
        self.processBlock(snd_block)

if __name__ == '__main__':
    audio = AudioHandler()
    total = 100
    for i in range(0, total):
        print('Recording block %i of %i...' % (i, total))
        audio.listen()
