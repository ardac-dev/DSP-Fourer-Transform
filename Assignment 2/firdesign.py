import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def design_fir_ifft(fs, stopbands_hz):
    """
    FIR design using IFFT of an ideal frequency response.

    fs : sampling rate (Hz)
    stopbands_hz : list of (f1, f2) frequency ranges to REMOVE (in Hz)
                   Example for 0.7–40 Hz band-pass:
                   stopbands_hz = [(0,0.7), (40, fs/2)]
    Returns:
        FIR coefficient array (1D)
    """

    L = 5001  # filter length (fixed, odd length gives linear phase)

    # 1. Construct ideal frequency response H[k]
    H = np.ones(L, dtype=float)
    freqs = np.fft.fftfreq(L, d=1.0/fs)

    # remove requested stopbands
    for f1, f2 in stopbands_hz:
        H[(np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)] = 0.0

    # also explicitly remove DC (edge case mentioned in assignment)
    H[np.isclose(freqs, 0.0, atol=1e-12)] = 0.0

    # 2. IFFT to obtain impulse response
    h = np.fft.ifft(H).real

    # 3. Shift to make filter causal (center → right)
    h = np.fft.ifftshift(h)

    # 4. Apply window to reduce ringing (allowed by assignment)
    #h = h * np.hamming(L)

    return h

fs = 500.0
stopbands = [(0, 0.7), (40, fs/2)]
coeffs = design_fir_ifft(fs, stopbands)

# Frequency response
Nresp = 16384
H = np.fft.rfft(coeffs, n=Nresp)
f = np.fft.rfftfreq(Nresp, d=1.0/fs)
mag = np.abs(H)

plt.figure()
plt.plot(f, mag)
plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H(f)| (linear)")
plt.title("FIR Frequency Response (Linear)")
plt.grid(True, alpha=0.3)
plt.axvspan(0.7, 40, alpha=0.15)
plt.axvline(0.7, linestyle="--", linewidth=1)
plt.axvline(40,  linestyle="--", linewidth=1)
plt.tight_layout()
plt.show()