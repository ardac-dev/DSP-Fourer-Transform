from firfilter import FIRfilter
from firdesign import design_fir_ifft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import iirnotch, filtfilt, butter

df = pd.read_csv('ecg.tsv', sep = "\t")

sample = df.iloc[:, 7]
num_samples = len(sample)

#for our noise source, we combined an unused channel with a 50 Hz sine wave 
noise = df.iloc[:, 8]

fs = 500
outputs = np.zeros(num_samples)
t = np.zeros(num_samples)
my_filter = FIRfilter(design_fir_ifft(fs, [[0,fs/2]]))

for i in range(num_samples):

    fifty_Hz = np.sin(2.0 * np.pi * 50 * i /fs) 

    outputs[i] = my_filter.doFilterAdaptive(sample[i], fifty_Hz + noise[i], 0.0001)

    t[i] = i/fs

plt.figure()
plt.plot(t, outputs)
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Heartrate Sample - Realtime Adaptive LMS FIR Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('lmsfirfilter - Time Domain Representation.svg')
plt.show()

#zoomed in plot for PQRST
plt.figure()
plt.plot(t, outputs)
plt.xlim([10,13])
plt.ylim([-0.001, 0.003])
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Heartrate Sample Zoomed in - Realtime Adaptive LMS FIR Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('lmsfirfilter - PQRST Integrity.svg')
plt.show()





