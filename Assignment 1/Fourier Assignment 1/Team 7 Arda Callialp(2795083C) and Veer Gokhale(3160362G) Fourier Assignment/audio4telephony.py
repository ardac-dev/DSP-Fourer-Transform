from scipy.io import wavfile
import numpy as np

# 1st load the wav file
sample_rate, data = wavfile.read("original_speech.wav")
x = data.astype(np.float32) / np.iinfo(data.dtype).max

# 2nd cutoff and target sampling rate
target_sr = 8000 # telecommunication voice standard 8kHz (8000 samples per second) human voice approx 300-3400 Hz Nyquis criteria: 3400*2=6800 to be safe 8000Hz
cutoff_hz  = 3400.0

num_samples = len(x)
fd_mag_raw = np.fft.fft(x)
k = np.arange(num_samples)
freq_raw = np.where(k < num_samples/2, k * sample_rate / num_samples, (k - num_samples) * sample_rate / num_samples)
fd_mag_raw[np.abs(freq_raw) >= cutoff_hz] = 0

# Back to time domain
x_lp = np.fft.ifft(fd_mag_raw).real

# 3rd downsample
N_old = len(x_lp)
t_old = np.arange(N_old) / sample_rate
T_end = (N_old - 1) / sample_rate
N_new = int(round(T_end * target_sr)) + 1
t_new = np.arange(N_new) / target_sr
y = np.interp(t_new, t_old, x_lp)

y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
out_path = "telephone_speech.wav"
wavfile.write(out_path, target_sr, y_int16)