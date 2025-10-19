from scipy.io import wavfile
import numpy as np

# 1) load the wav file
sr, data = wavfile.read("original_speech.wav")
x = data.astype(np.float32) / np.iinfo(data.dtype).max

# 2) Cutoff and target sampling rate
target_sr = 8000 # telecommunication voice standard 8kHz (8000 samples per second) human voice approx 300-3400 Hz Nyquis criteria: 3400*2=6800 to be safe 8000Hz
cutoff_hz  = 3400.0

X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), 1/sr)

X[freqs > cutoff_hz] = 0

# Back to time domain
x_lp = np.fft.irfft(X)

# 3) Downsample
N_old = len(x_lp)
t_old = np.arange(N_old) / sr
T_end = (N_old - 1) / sr
N_new = int(round(T_end * target_sr)) + 1
t_new = np.arange(N_new) / target_sr
y = np.interp(t_new, t_old, x_lp)

y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
out_path = "telephone_speech.wav"
wavfile.write(out_path, target_sr, y_int16)