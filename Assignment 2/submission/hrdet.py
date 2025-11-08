from firfilter import FIRfilter
from firdesign import design_fir_ifft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ecg.tsv', sep = "\t")

sample = df.iloc[:, 7]

fs = 500

num_samples = len(sample)
pre_filtered = np.zeros(num_samples)
match_filtered = np.zeros(num_samples)
detections = np.zeros(num_samples)
heartrate = np.zeros(num_samples)

t = np.zeros(num_samples)

def wavelet(t, f0=2, sigma=0.01):
    return 0.002* np.cos(2*np.pi*f0*t) * np.exp(-(t**2)/(2*sigma**2)) 

coeffs = [wavelet(i/fs) for i in range(-230, 230)] #obtaining coefficients for wavelet 
coeffs = coeffs[::-1] #time reversing the coefficients

pre_filter = FIRfilter(design_fir_ifft(fs, [[0, 0.7], [40, fs/2]]))
matched_filter = FIRfilter(coeffs)
last_detection = 0

for i in range(num_samples):
    pre_filtered[i] = pre_filter.dofilter(sample[i])
    match_filtered[i] = (matched_filter.dofilter(pre_filtered[i]))**2

    if (i == 0):
        heartrate[i] = 0

    elif (match_filtered[i] > 0.1e-8 and i > 6 * fs and i - last_detection > 0.3*fs): #heuristic for removing bogus detections
        detections[i] = 1

         #calculating heartrate in bpm

        if (last_detection != 0):
            heartrate[i] = 60/( (i - last_detection)/fs ) 
        last_detection = i

    else:
        heartrate[i] = heartrate[i-1]
    t[i] = i/fs
    
plt.figure()
#plt.plot(t, pre_filtered, label ='Pre-filtered Signal')
plt.plot(t, heartrate)
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Momentary Heartrate (BPM)')
plt.xlabel('Time (s)')
plt.ylabel('Momentary Heartrate (BPM)')
plt.savefig('Momentary Heartrate (BPM).svg')
plt.show()

plt.figure()
#plt.plot(t, pre_filtered, label ='Pre-filtered Signal')
plt.plot(t, match_filtered)
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Signal after matched filtering and squaring')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('Signal after matched filtering and squaring.svg')
plt.show()

plt.figure()
#plt.plot(t, pre_filtered, label ='Pre-filtered Signal')
plt.plot(t, detections)
#plt.plot(t, sample - sample.mean())
plt.grid(True)
plt.title('Heartrate Detections')
plt.xlabel('Time (s)')
plt.ylabel('Detection')
plt.savefig('Heartrate Detections.svg')
plt.show()

plt.figure()
plt.plot([i for i in range(0,460)], pre_filtered[5022:5482], label = "Pre-filtered Signal")
plt.plot([i for i in range(0,460)], [wavelet(i/fs) for i in range(-230, 230)], label = "Wavelet")
plt.legend()
plt.title('Example heartbeat and Wavelet used for matched filtering')
plt.xlabel('Number of samples')
plt.ylabel('Amplitude')
plt.savefig('heartbeat + wavelet side by side.svg')
plt.show()