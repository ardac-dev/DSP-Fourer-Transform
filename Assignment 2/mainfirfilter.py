from firfilter import FIRfilter
from firdesign import design_fir_ifft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('VeerArda_boarddiagram3.tsv', sep = "\t")
#df = df.T
print(df.shape)

pos_col_num = 7
sample = df.iloc[:, pos_col_num]


fs = 500

num_samples = len(sample)
outputs = np.zeros_like(sample)

my_filter = FIRfilter(design_fir_ifft(fs, [(0,0.7), (40, fs/2)]))

for i in range(num_samples):

    outputs[i] = my_filter.dofilter(sample[i])


t = [i/fs for i in range(len(sample))]

plt.figure()
plt.plot(t, outputs)
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('heartrate sample - filtered')
plt.xlabel('Seconds')
plt.ylabel('Amplitude')
plt.show()








