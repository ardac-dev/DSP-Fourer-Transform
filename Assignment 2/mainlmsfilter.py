from firfilter import FIRfilter
from firdesign import design_fir_ifft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import iirnotch, filtfilt, butter

df = pd.read_csv('VeerArda_boarddiagram3.tsv', sep = "\t")

sample = df.iloc[:, 7]
print(sample.max(), sample.min())
num_samples = len(sample)
noise = df.iloc[:, 8]
print(noise.max(), noise.min())

fs = 500
outputs = np.zeros(num_samples)
t = np.zeros(num_samples)
my_filter = FIRfilter(design_fir_ifft(fs, [[0,fs/2]]))

for i in range(num_samples):

    fifty_Hz = np.sin(2.0 * np.pi * 50 * i /fs) 
    DC_Hz = np.sin(2.0 * np.pi *0.5 * i /fs) 

    outputs[i] = my_filter.doFilterAdaptive(sample[i], fifty_Hz + noise[i], 0.0001)

    t[i] = i/fs

plt.figure()
plt.plot(t, outputs)
plt.plot(t, sample)
plt.grid(True)
plt.title('heartrate sample - filtered')
plt.xlabel('Seconds')
plt.ylabel('Amplitude')
plt.show()




