#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import yaml
import traceback

import warnings

from sklearn.externals import joblib

import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

from identify import only_voice_range

warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

DATA_DIR = '../../data/audio'
SPEAKERS_FN = 'speakers.yaml'

# FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30 # supports 10, 20 and 30 (ms)
# CHUNK_SIZE = 1024
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # chunk to read
# CHUNK_BYTES = CHUNK_SIZE * 2 # 16bit = 2 bytes, PCM

speakers_data = yaml.load(open(os.path.join(DATA_DIR, SPEAKERS_FN))) or {}

target_speaker = 'speakerA'

# mdl = 'classifier.mdl'

def iter_corpus(only_label=None, as_binary=None, except_label=None):
    """
    Iterates over data, returning tuples of the form (label, [feature0, feature1, ...featureN])
    """
    for audio_fn, speaker_label in speakers_data.items():
        # print(audio_fn, speaker_label)
        
        if only_label is not None and speaker_label != only_label:
            continue
        
        if except_label is not None and speaker_label == except_label:
            continue
        
        #https://stackoverflow.com/a/44800492/247542
        #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html
        # sample_rate, samples = wavfile.read(os.path.join(DATA_DIR, audio_fn))
        first_len = None
        for sample in sf.blocks(os.path.join(DATA_DIR, audio_fn), blocksize=CHUNK_SIZE):
            
            # print('sample_rate:', sample_rate) # samples/second
            # print('sample.shape:', sample.shape)
            f, t, Sxx = signal.spectrogram(sample, RATE)

            # Limit frequencies to the human voice range.
            # fmin = 50 # Hz
            # fmax = 300 # Hz
            # freq_slice = np.where((f >= fmin) & (f <= fmax))
            # f = f[freq_slice]
            # Sxx = Sxx[freq_slice,:][0]
            f, Sxx = only_voice_range(f, Sxx)

            Sxx = Sxx.flatten()
            if first_len is None:
                first_len = len(Sxx)
            elif len(Sxx) != first_len:
                continue
            # print('Sxx.shape:', Sxx.shape, len(Sxx))
            # assert Sxx.shape == (258,)
            assert Sxx.shape == (8,)
            
            if as_binary is not None:
                speaker_label = speaker_label == as_binary
            
            yield Sxx, speaker_label # X, y

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [

    # GARBAGE
    # ('gaussian_1_rbf1', lambda: GaussianProcessClassifier(1.0 * RBF(1.0))),#too cpu intensive
    # ('gradboost', lambda: GradientBoostingClassifier()),#slow and shitty

    # BAD
    # ('svc', lambda: SVC()),
    # ('svc_linear_c0025', lambda: SVC(kernel="linear", C=0.025)),
    # ('svc_gamma_2_c1', lambda: SVC(gamma=2, C=1)),
    # ('dtc_d5', lambda: DecisionTreeClassifier(max_depth=5)),
    # ('rfc_d5_est10_mf1', lambda: RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    # ('mlp_a1', lambda: MLPClassifier(alpha=1)),
    # ('ada', lambda: AdaBoostClassifier()),
    # ('gaussnb', lambda: GaussianNB()),
    # ('quaddis', lambda: QuadraticDiscriminantAnalysis()),
    ('bagging', lambda: BaggingClassifier()),
    
    # >=60
    ('knn', lambda: KNeighborsClassifier()),
    ('knn_3', lambda: KNeighborsClassifier(3)),
    
    # >=70
    ('dtc', lambda: DecisionTreeClassifier()),
    ('rfc', lambda: RandomForestClassifier()),
    
    # >=80
    ('extratrees', lambda: ExtraTreesClassifier()),
    
    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier
    #Training=85, testing=45
    # ('voting', lambda: VotingClassifier(estimators=[('knn', KNeighborsClassifier()), ('dtc', DecisionTreeClassifier()), ('rfc', RandomForestClassifier())])),
    
    ('voting', lambda: VotingClassifier(estimators=[('extratrees', ExtraTreesClassifier()), ('dtc', DecisionTreeClassifier()), ('bagging', BaggingClassifier())])),
    
]

all_data = list(iter_corpus())
# all_data = list(iter_corpus(except_label=target_speaker))#TODO:remove
# all_data = list(iter_corpus(as_binary=target_speaker))
all_X = [X for X, y in all_data]
print('all_X:', len(all_X))
all_Y = [y for X, y in all_data]
print('all_Y:', len(all_Y))
# print('all_Y:', [int(_) for _ in all_Y])

all_speakerA = list(iter_corpus(only_label=target_speaker))
all_speakerA_X = [X for X, y in all_speakerA]
all_speakerA_Y = [y for X, y in all_speakerA]
# all_speakerA_Y = [True for X, y in all_speakerA]

except_speakerA = list(iter_corpus(except_label=target_speaker))
except_speakerA_X = [X for X, y in except_speakerA]
except_speakerA_Y = [y for X, y in except_speakerA]

X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=0.1, random_state=0)
# print('X_test:', len(X_test))
# print('y_test:', len(y_test))
# X_train = [_row.tolist() for _row in X_train]
# X_test = [_row.tolist() for _row in X_test]

trained_classifiers = {} # {name: clf}
scores = {}
success_count = 0
for name, clf_cls in classifiers:
    try:
        print('Fitting:', name)
        clf = clf_cls()
        clf.fit(all_X, all_Y)
        training_score = clf.score(all_X, all_Y)
        print('\tTraining score:', training_score)
        trained_classifiers[name] = clf

        clf = clf_cls()
        clf.fit(X_train, y_train)
        testing_score = clf.score(X_test, y_test)
        print('\tTesting Score:', testing_score)
        
        speakerA_score = clf.score(all_speakerA_X, all_speakerA_Y)
        print('\tSpeakerA Score:', speakerA_score)
        
        not_speaker_score = 0.
        for X in except_speakerA_X:
            X = X.reshape(1, -1)
            y = clf.predict(X)
            # print('y:', y)
            not_speaker_score += y[0] != target_speaker
        not_speaker_score = not_speaker_score/len(except_speakerA_X)
        print('\t!SpeakerA Score:', not_speaker_score)
        
        total_score = speakerA_score * not_speaker_score
        scores[name] = total_score
        
        success_count += 1
    except Exception as exc:
        traceback.print_exc()
        raise

print('-'*80)
print('success_count:', success_count)
final_name = None
for final_name, score in sorted(scores.items(), key=lambda o: o[1]):
    print('%.06f' % score, final_name)

if final_name:
    final_clf = trained_classifiers.get(final_name)
    model_fn = 'identify_model.best.pkl'
    joblib.dump(final_clf, model_fn)
    print('Saved:', model_fn)
