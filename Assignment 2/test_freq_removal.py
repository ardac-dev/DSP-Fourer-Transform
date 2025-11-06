import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('VeerArda_boarddiagram3.tsv', sep="\t")
print(df.shape)

pos_col_num = 7  # 0-based index; gerekirse ayarla
sample = pd.to_numeric(df.iloc[:, pos_col_num], errors='coerce').to_numpy()

# DC offset'i kaldır
sample_norm = sample - np.mean(sample)

fs = 500.0
N  = len(sample_norm)
t  = np.arange(N)/fs

# --- FFT ---
X = np.fft.fft(sample_norm)
k = np.arange(N)
freq = np.where(k < N/2, k*fs/N, (k - N)*fs/N)

# --- Frekans filtreleme ---
Xf = X.copy()

# (1) DC tamamen sil
Xf[np.isclose(freq, 0.0, atol=1e-12)] = 0.0

# (2) 45–55 Hz notchu (mains interference)
Xf[(np.abs(freq) >= 45) & (np.abs(freq) <= 55)] = 0.0

# (3) 0.4 Hz altındaki tüm bileşenleri kes (baseline wander removal)
Xf[np.abs(freq) < 0.4] = 0.0

# --- Ters FFT: zaman domeni ---
y = np.fft.ifft(Xf).real

# --- Frekans alanı pozitif taraf ---
pos = freq >= 0
f_pos = freq[pos]
mag_orig = (2.0/N) * np.abs(X[pos])
mag_filt = (2.0/N) * np.abs(Xf[pos])

plt.figure()
plt.plot(f_pos, mag_orig, label='Original')
plt.plot(f_pos, mag_filt, label='Filtered (DC + 45–55Hz + <0.4Hz removed)', alpha=0.9)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (linear)')
plt.title('Frequency Domain (Linear)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 60)
plt.legend()
plt.tight_layout()
plt.show()

# --- Zaman domeni ---
plt.figure()
plt.plot(t, sample_norm, label='Original', alpha=0.6)
plt.plot(t, y, label='Filtered', alpha=0.9)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
