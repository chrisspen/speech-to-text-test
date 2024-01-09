import logging
from math import exp
from time import time
import os
import struct
import socket

from tester import BaseTester, RATE16K_MONO_WAV

def likelyhood_to_certainty(v):
    try:
        odds = exp(v)
        certainty = odds / (1. + odds)
    except OverflowError:
        certainty = 1.0
    return certainty

host_ip = "localhost"
host_port = 8086

def get_transcription(fn):
    fn = os.path.abspath(fn)
    assert fn and os.path.isfile(fn)
    print('Sending:', fn)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host_ip, host_port))
    sent = fn
    start = struct.pack('>i', len(sent))
    print('a:', start, type(start))
    sent = sent.encode('utf-8')
    print('b:', sent, type(sent))
    sock.sendall(start + sent)
    print('Speech[length=%d] Sent.' % len(sent))
    # Receive data from the server and shut down
    received = sock.recv(1024)
    received = received.decode('utf-8')
    print("Recognition Results: {}".format(received))
    sock.close()
    return received

class Tester(BaseTester):

    name = 'Paddle'

    audio_format = RATE16K_MONO_WAV

    def audio_to_text(self, fn):
        time_start = time()
        text = get_transcription(fn)
        td = time() - time_start
        print('text:', text)
        certainty = 1.0
        # print('certainty:', certainty)
        logging.debug("%s decoding took %8.2fs, certainty: %f", fn, td, certainty)
        # print(text, certainty)
        return text

if __name__ == '__main__':
    tester = Tester()
    tester.test()
