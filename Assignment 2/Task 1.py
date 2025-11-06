import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('VeerArda_boarddiagram3.tsv', sep = "\t")
#df = df.T
print(df.shape)

pos_col_num = 7
sample = df.iloc[:, pos_col_num]
sample_normalized = sample/max(abs(sample))
num_samples = len(sample)
fs = 500

x = [i/fs for i in range(len(sample))]

fd_mag_raw = np.fft.fft(sample_normalized)
k = np.arange(num_samples)
freq_raw = np.where(k < len(sample)/2, k*fs/len(sample), (k-len(sample))*fs/len(sample))
pos_mask = freq_raw >= 0

fd_db = 20*np.log10(2/len(sample) * np.abs(fd_mag_raw))[pos_mask]
freq = freq_raw[pos_mask]

plt.plot(freq, fd_db)
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.title('Frequency Domain Representation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim([0.1,100])
plt.show()


#plotting a single sample
plt.plot(x, sample)
plt.title('heartrate sample')
plt.xlabel('Seconds')
plt.ylabel('Amplitude')
plt.show()

