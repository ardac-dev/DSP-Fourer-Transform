import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ecg.tsv', sep = "\t")
#df = df.T

sample = df.iloc[:, 7]
num_samples = len(sample)
fs = 500

x = [i/fs for i in range(len(sample))]

fd_mag_raw = np.fft.fft(sample)
k = np.arange(num_samples)
freq_raw = np.where(k < len(sample)/2, k*fs/len(sample), (k-len(sample))*fs/len(sample))
pos_mask = freq_raw >= 0
fd_db = 20*np.log10(2/len(sample) * np.abs(fd_mag_raw))[pos_mask]
freq = freq_raw[pos_mask]

plt.plot(freq, fd_db)
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.title('Frequency Domain Representation')
plt.grid(True)
plt.tight_layout()
plt.savefig('Task 1 - Frequency Domain Representation.svg')
plt.show()


#plotting a single sample
plt.plot(x, sample)
plt.title('Heartrate Sample')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('Task 1 - Time Domain Representation.svg')
plt.show()

