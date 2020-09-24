import os
import logging
from time import time
from subprocess import getoutput

from tester import BaseTester, RATE16K_MONO_WAV


class Tester(BaseTester):

    name = 'Cheetah'

    audio_format = RATE16K_MONO_WAV

    def audio_to_text(self, fn):
        time_start = time()

        fn = os.path.abspath(fn)
        text = getoutput(f'cd /home/chris/git/cheetah; .env/bin/python demo/python/cheetah_demo.py --audio_paths {fn}').split('\n')[0].strip().lower()

        td = time() - time_start
        print('text:[%s]' % text)
        certainty = 1.0
        # print('certainty:', certainty)
        logging.debug("%s decoding took %8.2fs, certainty: %f", fn, td, certainty)
        # print(text, certainty)
        return text


if __name__ == '__main__':
    tester = Tester()
    tester.test()
