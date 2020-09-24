# import sys
# import argparse
import math
import logging
from math import exp
from time import time

import numpy as np
import h5py
import scipy.io.wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from tester import BaseTester, RATE16K_MONO_WAV

# pylint: disable=redefined-builtin

def likelyhood_to_certainty(v):
    try:
        odds = exp(v)
        certainty = odds / (1. + odds)
    except OverflowError:
        certainty = 1.0
    return certainty


def load_model_en_jasper(model_weights, batch_norm_eps = 0.001, num_classes = 29, ABC = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'|"):
    class JasperNet(nn.ModuleList):
        def __init__(self, num_classes):
            def conv_bn_residual(kernel_size, num_channels, stride = 1, dilation = 1, padding = 0, repeat = 1, num_channels_residual=None):
                if num_channels_residual is None:
                    num_channels_residual = []
                return nn.ModuleDict(dict(
                    conv = nn.ModuleList([
                        nn.Conv1d(num_channels[0] if i == 0 else num_channels[1],
                        num_channels[1],
                        kernel_size = kernel_size,
                        stride = stride,
                        dilation = dilation, padding = padding) for i in range(repeat)]),
                    conv_residual = nn.ModuleList([nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual])
                ))

            blocks = nn.ModuleList([
                conv_bn_residual(kernel_size = 11, num_channels = (64, 256), padding = 5, stride = 2),

                conv_bn_residual(kernel_size = 11, num_channels = (256, 256), padding = 5, repeat = 5, num_channels_residual = [256]),
                conv_bn_residual(kernel_size = 11, num_channels = (256, 256), padding = 5, repeat = 5, num_channels_residual = [256, 256]),

                conv_bn_residual(kernel_size = 13, num_channels = (256, 384), padding = 6, repeat = 5, num_channels_residual = [256, 256, 256]),
                conv_bn_residual(kernel_size = 13, num_channels = (384, 384), padding = 6, repeat = 5, num_channels_residual = [256, 256, 256, 384]),

                conv_bn_residual(kernel_size = 17, num_channels = (384, 512), padding = 8, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384]),
                conv_bn_residual(
                    kernel_size = 17, num_channels = (512, 512), padding = 8, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512]),

                conv_bn_residual(
                    kernel_size = 21, num_channels = (512, 640), padding = 10, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512]),
                conv_bn_residual(
                    kernel_size = 21, num_channels = (640, 640), padding = 10, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640]),

                conv_bn_residual(
                    kernel_size = 25,
                    num_channels = (640, 768), padding = 12, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640]),
                conv_bn_residual(
                    kernel_size = 25,
                    num_channels = (768, 768), padding = 12, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640, 768]),

                conv_bn_residual(kernel_size = 29, num_channels = (768, 896), padding = 28, dilation = 2),
                conv_bn_residual(kernel_size = 1, num_channels = (896, 1024)),

                nn.Conv1d(1024, num_classes, 1)
            ])
            super(JasperNet, self).__init__(blocks)

        def forward(self, x):
            residual = []
            for i, block in enumerate(list(self)[:-1]):
                for conv in block.conv[:-1]:
                    x = F.relu(conv(x), inplace = True)
                x = block.conv[-1](x)
                for conv, r in zip(block.conv_residual, residual if i < len(self) - 3 else []):
                    x = x + conv(r)
                x = F.relu(x, inplace = True)
                residual.append(x)
            return self[-1](x)

    model = JasperNet(num_classes = len(ABC))
    h = h5py.File(model_weights)
    to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
    state_dict = {}
    for param_name, param in model.state_dict().items():
        ij = [int(c) for c in param_name.split('.') if c.isdigit()]
        weight, bias = None, None
        if len(ij) > 1:
            weight, moving_mean, moving_variance, gamma, beta = [
                to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}{1 + ij[1]}/{suffix}') \
                for suffix in ['kernel', 'bn/moving_mean', 'bn/moving_variance', 'bn/gamma', 'bn/beta']] \
                if 'residual' not in param_name else [to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}5/{suffix}') \
                for suffix in [
                    f'res_{ij[1]}/kernel', f'res_bn_{ij[1]}/moving_mean', f'res_bn_{ij[1]}/moving_variance', f'res_bn_{ij[1]}/gamma', f'res_bn_{ij[1]}/beta']]
            weight, bias = fuse_conv_bn(weight.permute(2, 1, 0), moving_mean, moving_variance, gamma, beta, batch_norm_eps = batch_norm_eps)
        else:
            weight, bias = [to_tensor(f'ForwardPass/fully_connected_ctc_decoder/fully_connected/{suffix}') for suffix in ['kernel', 'bias']]
            weight = weight.t().unsqueeze(-1)

        state_dict[param_name] = (weight if 'weight' in param_name else bias).to(param.dtype)
    model.load_state_dict(state_dict)

    def frontend(signal, sample_freq, window_size=20e-3, window_stride=10e-3, dither = 1e-5, window_fn = np.hanning, num_features = 64):
        def get_melscale_filterbanks(sr, n_fft, n_mels, fmin, fmax, dtype = np.float32):
            def hz_to_mel(frequencies):
                frequencies = np.asanyarray(frequencies)
                f_min = 0.0
                f_sp = 200.0 / 3
                mels = (frequencies - f_min) / f_sp
                min_log_hz = 1000.0
                min_log_mel = (min_log_hz - f_min) / f_sp
                logstep = np.log(6.4) / 27.0

                if frequencies.ndim:
                    log_t = (frequencies >= min_log_hz)
                    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
                elif frequencies >= min_log_hz:
                    mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

                return mels

            def mel_to_hz(mels):
                mels = np.asanyarray(mels)
                f_min = 0.0
                f_sp = 200.0 / 3
                freqs = f_min + f_sp * mels
                min_log_hz = 1000.0
                min_log_mel = (min_log_hz - f_min) / f_sp
                logstep = np.log(6.4) / 27.0

                if mels.ndim:
                    log_t = (mels >= min_log_mel)
                    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
                elif mels >= min_log_mel:
                    freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

                return freqs

            n_mels = int(n_mels)
            weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

            fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft//2),endpoint=True)
            mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))

            fdiff = np.diff(mel_f)
            ramps = np.subtract.outer(mel_f, fftfreqs)

            for i in range(n_mels):
                lower = -ramps[i] / fdiff[i]
                upper = ramps[i+2] / fdiff[i+1]
                weights[i] = np.maximum(0, np.minimum(lower, upper))

            enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
            return torch.from_numpy(weights)

        signal = signal / (signal.abs().max() + 1e-5)
        audio_duration = len(signal) * 1.0 / sample_freq
        n_window_size = int(sample_freq * window_size)
        n_window_stride = int(sample_freq * window_stride)
        num_fft = 2**math.ceil(math.log2(window_size*sample_freq))

        signal += dither * torch.randn_like(signal)
        S = torch.stft(
            signal,
            num_fft,
            hop_length=int(window_stride * sample_freq),
            win_length=int(window_size * sample_freq),
            window=torch.hann_window(int(window_size * sample_freq)).type_as(signal), pad_mode = 'reflect', center = True).pow(2).sum(dim = -1)
        mel_basis = get_melscale_filterbanks(sample_freq, num_fft, num_features, fmin=0, fmax=int(sample_freq/2)).type_as(S)

        features = torch.log(torch.matmul(mel_basis, S) + 1e-20)
        mean = features.mean(dim=1, keepdim=True)
        std_dev = features.std(dim=1, keepdim=True)
        return (features - mean) / std_dev

    return frontend, model, (lambda c: ABC[c]), ABC.index

def load_model_ru_w2l(model_weights, batch_norm_eps=1e-05, ABC='|АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ2* '):
    def conv_bn(kernel_size, num_channels, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    model = nn.Sequential(
        conv_bn(kernel_size = 13, num_channels = (161, 768), stride = 2, padding = 6),
        conv_bn(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
        conv_bn(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
        conv_bn(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
        conv_bn(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
        conv_bn(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
        conv_bn(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
        conv_bn(kernel_size = 31, num_channels = (768, 2048), stride = 1, padding = 15),
        conv_bn(kernel_size = 1,  num_channels = (2048, 2048), stride = 1, padding = 0),
        nn.Conv1d(2048, len(ABC), kernel_size=1, stride=1)
    )

    h = h5py.File(model_weights)
    to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
    state_dict = {}
    for param_name, param in model.state_dict().items():
        ij = [int(c) for c in param_name if c.isdigit()]
        if len(ij) > 1:
            weight, moving_mean, moving_variance, gamma, beta = [
                to_tensor(f'rnns.{ij[0] * 3}.weight')] + [to_tensor(f'rnns.{ij[0] * 3 + 1}.{suffix}') \
                for suffix in ['running_mean', 'running_var', 'weight', 'bias']]
            weight, bias = fuse_conv_bn(weight, moving_mean, moving_variance, gamma, beta, batch_norm_eps = batch_norm_eps)
        else:
            weight, bias = [to_tensor(f'fc.0.{suffix}') for suffix in ['weight', 'bias']]
        state_dict[param_name] = (weight if 'weight' in param_name else bias).to(param.dtype)
    model.load_state_dict(state_dict)

    def frontend(signal, sample_rate, window_size = 0.020, window_stride = 0.010, window = 'hann'):
        signal = signal / signal.abs().max()
        if sample_rate == 8000:
            signal, sample_rate = F.interpolate(signal.view(1, 1, -1), scale_factor = 2).squeeze(), 16000
        win_length = int(sample_rate * (window_size + 1e-8))
        hop_length = int(sample_rate * (window_stride + 1e-8))
        nfft = win_length
        return torch.stft(
            signal,
            nfft,
            win_length = win_length,
            hop_length = hop_length,
            window = torch.hann_window(nfft).type_as(signal), pad_mode = 'reflect', center = True).pow(2).sum(dim = -1).add(1e-9).sqrt()

    return frontend, model, (lambda c: ABC[c]), ABC.index

def load_model_en_w2l(model_weights, batch_norm_eps = 0.001, ABC = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'|"):
    def conv_block(kernel_size, num_channels, stride = 1, dilation = 1, repeat = 1, padding = 0):
        modules = []
        for i in range(repeat):
            modules.append(nn.Conv1d(
                num_channels[0] if i == 0 else num_channels[1],
                num_channels[1],
                kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding))
            modules.append(nn.Hardtanh(0, 20, inplace = True))
        return nn.Sequential(*modules)

    model = nn.Sequential(
        conv_block(kernel_size = 11, num_channels = (64, 256), stride = 2, padding = 5),
        conv_block(kernel_size = 11, num_channels = (256, 256), repeat = 3, padding = 5),
        conv_block(kernel_size = 13, num_channels = (256, 384), repeat = 3, padding = 6),
        conv_block(kernel_size = 17, num_channels = (384, 512), repeat = 3, padding = 8),
        conv_block(kernel_size = 21, num_channels = (512, 640), repeat = 3, padding = 10),
        conv_block(kernel_size = 25, num_channels = (640, 768), repeat = 3, padding = 12),
        conv_block(kernel_size = 29, num_channels = (768, 896), repeat = 1, padding = 28, dilation = 2),
        conv_block(kernel_size = 1, num_channels = (896, 1024), repeat = 1),
        nn.Conv1d(1024, len(ABC), 1)
    )

    h = h5py.File(model_weights)
    to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
    state_dict = {}
    for param_name, param in model.state_dict().items():
        ij = [int(c) for c in param_name if c.isdigit()]
        if len(ij) > 1:
            weight, moving_mean, moving_variance, gamma, beta = [
                to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}{1 + ij[1] // 2}/{suffix}')
                for suffix in ['kernel', 'bn/moving_mean', 'bn/moving_variance', 'bn/gamma', 'bn/beta']]
            weight, bias = fuse_conv_bn(weight.permute(2, 1, 0), moving_mean, moving_variance, gamma, beta, batch_norm_eps = batch_norm_eps)
        else:
            weight, bias = [to_tensor(f'ForwardPass/fully_connected_ctc_decoder/fully_connected/{suffix}') for suffix in ['kernel', 'bias']]
            weight = weight.t().unsqueeze(-1)
        state_dict[param_name] = (weight if 'weight' in param_name else bias).to(param.dtype)
    model.load_state_dict(state_dict)

    def frontend(signal, sample_rate, nfft = 512, nfilt = 64, preemph = 0.97, window_size = 0.020, window_stride = 0.010):
        def get_melscale_filterbanks(nfilt, nfft, samplerate):
            hz2mel = lambda hz: 2595 * math.log10(1+hz/700.)
            mel2hz = lambda mel: torch.mul(700, torch.sub(torch.pow(10, torch.div(mel, 2595)), 1))
            lowfreq = 0
            highfreq = samplerate // 2
            lowmel = hz2mel(lowfreq)
            highmel = hz2mel(highfreq)
            melpoints = torch.linspace(lowmel,highmel,nfilt+2)
            bin = torch.floor(torch.mul(nfft+1, torch.div(mel2hz(melpoints), samplerate))).tolist()

            fbank = torch.zeros([nfilt, nfft // 2 + 1]).tolist()
            for j in range(nfilt):
                for i in range(int(bin[j]), int(bin[j+1])):
                    fbank[j][i] = (i - bin[j]) / (bin[j+1]-bin[j])
                for i in range(int(bin[j+1]), int(bin[j+2])):
                    fbank[j][i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
            return torch.tensor(fbank)

        preemphasis = lambda signal, coeff: torch.cat([signal[:1], torch.sub(signal[1:], torch.mul(signal[:-1], coeff))])
        win_length = int(sample_rate * (window_size + 1e-8))
        hop_length = int(sample_rate * (window_stride + 1e-8))
        pspec = torch.stft(
            preemphasis(signal, preemph),
            nfft,
            win_length=win_length,
            hop_length=hop_length,
            window=torch.hann_window(win_length), pad_mode = 'constant', center = False).pow(2).sum(dim = -1) / nfft
        mel_basis = get_melscale_filterbanks(nfilt, nfft, sample_rate).type_as(pspec)
        features = torch.log(torch.add(torch.matmul(mel_basis, pspec), 1e-20))
        return (features - features.mean()) / features.std()

    return frontend, model, (lambda c: ABC[c]), ABC.index

def fuse_conv_bn(weight, moving_mean, moving_variance, gamma, beta, batch_norm_eps):
    factor = gamma * (moving_variance + batch_norm_eps).rsqrt()
    weight *= factor.view(-1, *([1] * (weight.dim() - 1)))
    bias = beta - moving_mean * factor
    return weight, bias

def decode_greedy(scores, idx2chr):
    decoded_greedy = scores.argmax(dim = 0).tolist()
    values, indices = scores.max(0)
    # print('values:', values, sum(values), sum(values)/len(values))
    # print('indices:', indices)
    avg_likelyhood = sum(values)/len(values)
    decoded_text = ''.join(map(idx2chr, decoded_greedy))
    text = ''.join(c for i, c in enumerate(decoded_text) if (i == 0 or c != decoded_text[i - 1]) and c != '|')
    certainty = likelyhood_to_certainty(avg_likelyhood)
    return text, certainty


class Tester(BaseTester):

    name = 'Wav2Letter'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

        model = 'en_w2l'
        weights = '../data/models/wav2letter/w2l_plus_large_mp.h5'

        self.device = 'cpu'

        torch.set_grad_enabled(False)
        self.frontend, self.model, self.idx2chr, self.chr2idx = load_model_en_w2l(weights)

    def audio_to_text(self, fn):
        time_start = time()

        sample_rate, signal = scipy.io.wavfile.read(fn)
        assert sample_rate in [8000, 16000]
        features = self.frontend(torch.from_numpy(signal).to(torch.float32), sample_rate)
        scores = self.model.to(self.device)(features.unsqueeze(0).to(self.device)).squeeze(0)
        text, certainty = decode_greedy(scores, self.idx2chr)
        td = time() - time_start
        print('text:', text)
        print('certainty:', certainty)
        logging.debug("%s decoding took %8.2fs, certainty: %f", fn, td, certainty)
        # print(text, certainty)
        return text


if __name__ == '__main__':
    tester = Tester()
    tester.test()
