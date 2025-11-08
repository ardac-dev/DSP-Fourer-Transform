from firfilter import FIRfilter
from firdesign import design_fir_ifft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ecg.tsv', sep = "\t")

sample = df.iloc[:, 7]

fs = 500

num_samples = len(sample)
outputs = np.zeros(num_samples)
t = np.zeros(num_samples)

myfilter = FIRfilter(design_fir_ifft(fs, [[0, 0.7], [40, fs/2]]))

for i in range(num_samples):
    outputs[i] = myfilter.dofilter(sample[i])
    t[i] = i/fs

#main plot
plt.figure()
plt.plot(t, outputs)
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Heartrate Sample - Realtime FIR Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('mainfirfilter - Time Domain Representation.svg')
plt.show()

#zoomed in plot for PQRST
plt.figure()
plt.plot(t, outputs)
plt.xlim([10,13])
plt.ylim([-0.001, 0.003])
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Heartrate Sample Zoomed in - Realtime FIR Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('mainfirfilter - PQRST Integrity.svg')
plt.show()
























