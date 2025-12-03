import numpy as np
import matplotlib.pyplot as plt

from lpf_IIR import IIRlpf
from hpf_IIR import IIRhpf

# ============================================
# PARAMETERS
# ============================================
FS = 1000.0          # sampling frequency (Hz)
N  = 16384           # number of samples

LPF_CUTOFF = 300.0    # Hz
HPF_CUTOFF = 80.0   # Hz

# ============================================
# 1) GENERATE WHITE NOISE TEST SIGNAL
# ============================================
np.random.seed(0)
x = np.random.randn(N)

# ============================================
# 2) SET UP 3 CASCADED LPFs AND 3 CASCADED HPFs
# ============================================
lpf_sections = [IIRlpf(LPF_CUTOFF, FS) for _ in range(3)]
hpf_sections = [IIRhpf(HPF_CUTOFF, FS) for _ in range(3)]

for sec in lpf_sections:
    sec.calc_coeffs()
for sec in hpf_sections:
    sec.calc_coeffs()

# ============================================
# 3) FILTER NOISE SAMPLE-BY-SAMPLE THROUGH CHAINS
# ============================================
y_lpf = np.zeros(N)
y_hpf = np.zeros(N)

for n in range(N):
    y = x[n]
    # 3 cascaded LPF sections
    for sec in lpf_sections:
        y = sec.dofilter(y)
    y_lpf[n] = y

for n in range(N):
    y = x[n]
    # 3 cascaded HPF sections
    for sec in hpf_sections:
        y = sec.dofilter(y)
    y_hpf[n] = y

# ============================================
# 4) FFT BEFORE / AFTER FILTERING
# ============================================
window = np.hanning(N)

X      = np.fft.rfft(x * window)
Y_LPF  = np.fft.rfft(y_lpf * window)
Y_HPF  = np.fft.rfft(y_hpf * window)
freqs  = np.fft.rfftfreq(N, d=1.0 / FS)

eps = 1e-12
ref = np.max(np.abs(X)) + eps

X_mag     = 20 * np.log10(np.abs(X)     / ref)
Y_LPF_mag = 20 * np.log10(np.abs(Y_LPF) / ref)
Y_HPF_mag = 20 * np.log10(np.abs(Y_HPF) / ref)

# ============================================
# 5) PLOT RESULTS
# ============================================
plt.figure(figsize=(10, 8))

# ----- LPF chain -----
plt.subplot(2, 1, 1)
plt.plot(freqs, X_mag,     label='Input noise')
plt.plot(freqs, Y_LPF_mag, label=f'3× LPF output (fc={LPF_CUTOFF} Hz)')
plt.title('6th-order (3×2nd-order) Butterworth LPF – Noise Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB, normalized)')
plt.grid(True)
plt.legend()
plt.xlim(0, FS / 2)

# ----- HPF chain -----
plt.subplot(2, 1, 2)
plt.plot(freqs, X_mag,     label='Input noise')
plt.plot(freqs, Y_HPF_mag, label=f'3× HPF output (fc={HPF_CUTOFF} Hz)')
plt.title('6th-order (3×2nd-order) Butterworth HPF – Noise Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB, normalized)')
plt.grid(True)
plt.legend()
plt.xlim(0, FS / 2)

plt.tight_layout()
plt.show()

