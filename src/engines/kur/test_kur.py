import sys
import os

from kur.kurfile import Kurfile
from kur.engine import JinjaEngine
from kur.model.hooks import TranscriptHook
from kur.utils import Normalize, get_audio_features
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    model, norm, trans, rev, blank = load()

    for root, dirs, files in os.walk(sys.argv[1]):
        for filename in tqdm(files):
            if not filename.endswith('.wav'):
                continue
            file_path = os.path.join(root, filename)
            if os.path.exists(file_path):
                continue
            try:
                outputs, feats = get_output(file_path, norm, model)
            except OSError:
                continue
            text = trans.argmax_decode(outputs, rev, blank)
            print('====={}:\n{}\n'.format(filename, text))
            plot(outputs, rev, blank)


def load():
    spec_file = 'speech.yml'
    w_file = 'weights'
    spec = Kurfile(spec_file, JinjaEngine())
    spec.parse()

    model = spec.get_model()
    model.backend.compile(model)
    model.restore(w_file)

    norm = Normalize(center=True, scale=True, rotate=True)
    norm.restore('norm.yml')

    trans = TranscriptHook()
    rev = {0: ' ', 1: "'", 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o',
        17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
    blank = 28
    return model, norm, trans, rev, blank


def plot(outputs, rev, blank, title=''):
    fig, ax = plt.subplots()
    plt.imshow(outputs.T, aspect='auto')
    ax.set_yticks(list(rev.keys()) + [blank])
    ax.set_yticklabels(list(rev.values()) + ['null'])
    plt.grid(True)
    plt.title(title)
    plt.show()


def get_output(file_path, norm, model):
    feats = get_audio_features(file_path, 'spec', high_freq=8000)
    inputs = norm.apply(feats)
    pdf, _ = model.backend.evaluate(model, data={'utterance': inputs[np.newaxis, :, :]})
    return pdf['asr'].squeeze(), feats



if __name__ == "__main__":
    main()
