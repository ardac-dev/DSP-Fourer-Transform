import time
from lpf_IIR import IIRlpf
from hpf_IIR import IIRhpf
import pyfirmata2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
FS = 1000.0                 # sampling frequency (Hz)
DURATION = 5.0              # seconds
N_SAMPLES = int(FS * DURATION)

HPF_CUTOFF = 70            # Hz
LPF_CUTOFF = 300.0          # Hz

IGNORE_LOW_FREQ = 10.0      # Hz threshold to ignore

# ============================================================
# SET UP FILTERS (3× HPF + 3× LPF CASCADED)
# ============================================================
hpf_sections = [IIRhpf(HPF_CUTOFF, FS) for _ in range(3)]
lpf_sections = [IIRlpf(LPF_CUTOFF, FS) for _ in range(3)]

for sec in hpf_sections:
    sec.calc_coeffs()
for sec in lpf_sections:
    sec.calc_coeffs()

def bandpass_filter_sample(x):
    """Apply 3× HPF then 3× LPF to one sample."""
    y = x
    for sec in hpf_sections:
        y = sec.dofilter(y)
    for sec in lpf_sections:
        y = sec.dofilter(y)
    return y

# ============================================================
# CONNECT TO ARDUINO
# ============================================================
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)   # 1 kHz

raw_samples = []
filtered_samples = []

def a0_callback(data):
    if data is None:
        return
    if len(raw_samples) >= N_SAMPLES:
        return

    v = data * 5.0
    y = bandpass_filter_sample(v)

    raw_samples.append(v)
    filtered_samples.append(y)

a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print(f"Collecting {DURATION} seconds of data...")

while len(raw_samples) < N_SAMPLES:
    time.sleep(0.01)

print("Done. Stopping board.")
a0.disable_reporting()
board.exit()

# ============================================================
# FFT (LINEAR SCALE, IGNORE 0–10 Hz)
# ============================================================
raw = np.array(raw_samples)
filt = np.array(filtered_samples)

window = np.hanning(N_SAMPLES)
raw_w  = raw * window
filt_w = filt * window

RAW_FFT  = np.abs(np.fft.rfft(raw_w))
FILT_FFT = np.abs(np.fft.rfft(filt_w))
freqs    = np.fft.rfftfreq(N_SAMPLES, d=1.0 / FS)

# --------- ZERO OUT 0–10 Hz ---------
mask = freqs >= IGNORE_LOW_FREQ
RAW_FFT  = RAW_FFT  * mask
FILT_FFT = FILT_FFT * mask

# Normalize to max of raw spectrum
ref = np.max(RAW_FFT) + 1e-12
RAW_FFT  /= ref
FILT_FFT /= ref

# ============================================================
# PLOTS — LINEAR SCALE
# ============================================================
plt.figure(figsize=(10, 8))

# ----- Time domain -----
plt.subplot(2, 1, 1)
t = np.arange(N_SAMPLES) / FS
plt.plot(t, raw, label="Raw", alpha=0.5)
plt.plot(t, filt, label="Filtered", alpha=0.8)
plt.xlim(0, DURATION)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("Time-domain (5 s capture)")
plt.grid(True)
plt.legend()

# ----- Frequency domain (linear) -----
plt.subplot(2, 1, 2)
plt.plot(freqs, RAW_FFT,  label="Raw spectrum (≥10 Hz)")
plt.plot(freqs, FILT_FFT, label="Filtered spectrum (≥10 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude (linear)")
plt.title("Frequency-domain behaviour (0–10 Hz removed)")
plt.grid(True)
plt.legend()
plt.xlim(0, FS/2)

plt.tight_layout()
plt.show()