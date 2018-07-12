import numpy as np

def only_voice_range(f, Sxx, fmin=50, fmax=300):
    # Limit frequencies to the human voice range.
    freq_slice = np.where((f >= fmin) & (f <= fmax))
    f = f[freq_slice]
    Sxx = Sxx[freq_slice,:][0]
    return f, Sxx
