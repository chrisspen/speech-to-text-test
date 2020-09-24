import sys
import logging
import warnings
from time import time

from tester import BaseTester, RATE16K_MONO_WAV

sys.path.insert(0, '/home/chris/git/deepspeech.pytorch')

from utils import load_model # pylint: disable=wrong-import-position

warnings.simplefilter('ignore')

from decoder import GreedyDecoder # pylint: disable=wrong-import-position

import torch # pylint: disable=wrong-import-position

from data.data_loader import SpectrogramParser # pylint: disable=wrong-import-position


def decode_results(decoded_output, decoded_offsets, top_paths=1):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                # "name": os.path.basename(args.model_path)
            },
            "language_model": {
                # "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                # "lm": args.lm_path is not None,
                # "alpha": args.alpha if args.lm_path is not None else None,
                # "beta": args.beta if args.lm_path is not None else None,
                # "type": args.decoder,
            }
        }
    }
    for b, _b in enumerate(decoded_output):
        for pi in range(min(top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            results['output'].append(result)
    return results


def transcribe(audio_path, spect_parser, model, decoder, device, use_half):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    # return decoded_output, decoded_offsets
    text = decode_results(decoded_output, decoded_offsets)['output'][0]['transcription']
    return text


class Tester(BaseTester):

    name = 'Kaldi'

    audio_format = RATE16K_MONO_WAV

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        device = 'cpu'
        model_path = '/home/chris/git/deepspeech.pytorch/models/ted_pretrained_v2.pth'
        half = False
        model = load_model(device, model_path, half)

        # if args.decoder == "beam":
            # from decoder import BeamCTCDecoder
            # decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                     # cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                     # beam_width=args.beam_width, num_processes=args.lm_workers)
        # else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

        self.half = half
        self.device = device
        self.decoder = decoder
        self.model = model
        self.spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

    def audio_to_text(self, fn):
        time_start = time()

        # sample_rate, signal = scipy.io.wavfile.read(fn)
        # assert sample_rate in [8000, 16000]
        # features = self.frontend(torch.from_numpy(signal).to(torch.float32), sample_rate)
        # scores = self.model.to(self.device)(features.unsqueeze(0).to(self.device)).squeeze(0)
        # text, certainty = decode_greedy(scores, self.idx2chr)

        text = transcribe(audio_path=fn,
             spect_parser=self.spect_parser,
             model=self.model,
             decoder=self.decoder,
             device=self.device,
             use_half=self.half)
        # print(json.dumps(decode_results(decoded_output, decoded_offsets)))
        # print(decode_results(decoded_output, decoded_offsets)['output'][0]['transcription'])

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
