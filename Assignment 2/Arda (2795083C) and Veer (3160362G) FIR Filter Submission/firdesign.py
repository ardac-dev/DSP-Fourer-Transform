import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def design_fir_ifft(fs, stopbands_hz):

    L = 5001  # filter length (must be odd)

    X = np.ones(L, dtype=float)

    freqs_pos = np.arange(0,250.1, 0.1)
    freqs_neg = np.arange(-250, 0, 0.1)
    freqs = np.concatenate((freqs_pos, freqs_neg))


    # remove requested stopbands
    for f1, f2 in stopbands_hz:
        X[(np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)] = 0.0

    #IFFT to obtain impulse response
    x = np.fft.ifft(X).real
    t = np.arange(L)/fs

    #shifting to center at L/2
    h = np.zeros(L)
    h[0:L//2 + 1] = x[L//2 : L]
    h[L//2 + 1 :] = x[0 : L//2]

    # 4. Apply window to reduce ringing (allowed by assignment)
    h = h * np.hamming(L)

    return h

