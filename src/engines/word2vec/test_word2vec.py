#!/usr/bin/env python
import torch
import librosa
from fairseq.models.wav2vec import Wav2VecModel

cp = torch.load('../data/models/word2vec/wav2vec_large.pt', map_location=torch.device('cpu'))
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

wave_file_path = '../data/audio/go-forward-two-meters-and-then-stop.rate16k-mono.wav'
# wav_input = librosa.load(wave_file_path)
signal, sr = librosa.load(wave_file_path)
tensors = torch.from_numpy(signal)
z = model.feature_extractor(tensors)
c = model.feature_aggregator(z)
print('c:', c)
