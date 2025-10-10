# lab1_task2_downsample.py
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly
import numpy as np
import matplotlib.pyplot as plt

# 1) WAV yükle
in_path = "DSP_Assigment_1_Q1.wav"
sr, data = wavfile.read(in_path)

# Mono'ya indir (stereo ise sol kanal)
if data.ndim > 1:
    data = data[:, 0]

# Float'a çevir / normalize [-1, 1]
x = data.astype(np.float32) / 32768.0

# 2) Cutoff (telephony: 8 kHz aim → cutoff ≈ 3.4 kHz)
target_sr = 4000 # telecominication voice standard 8kHz (8000 samples per second) human voice approx 300-3400 Hz Nyquis criteria: 3400*2=6800 to be safe 8000Hz
cutoff_hz  = 2000.0 # not 3400 because cutoff = 0.8 * (target_sr/2) = 0.8*4000 = 3200

X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), 1/sr)

# Zero out everything above 4.4 kHz
X[freqs > target_sr] = 0

# Back to time domain
x_lp = np.fft.irfft(X)

# 3) Downsample (resample_poly)
from math import gcd
g = gcd(sr, target_sr)
up = target_sr // g
down = sr // g
y = resample_poly(x_lp, up, down)

# 4) 16-bit PCM'e çevirip TEK dosya yaz
y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
out_path = "speech_telephony_8k.wav"
wavfile.write(out_path, target_sr, y_int16)

print(f"Saved: {out_path} | from {sr} Hz → {target_sr} Hz, cutoff={cutoff_hz} Hz")