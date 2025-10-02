from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

sample_rate, data = wavfile.read("DSP_Assigment_1_Q1.wav")

print("Sample rate:", sample_rate)
print("Data shape:", data.shape)
print("Data type:", data.dtype)

if data.ndim > 1: # if its over one its sterio 
    data = data[:, 0] # if its sterio then discarding right channel and only looking at the left channel
    
y = data.astype(np.float32) / 32768.0 #normalizing the signal to be in between -1 and 1 range

t = np.arange(len(y)) / sample_rate #creating a time array for each sample
duration = len(y) / sample_rate # finding the total duration of the recording

#plotting
plt.figure(figsize=(10,3))
plt.plot(t, y, linewidth=0.8) # x-axis = (t) time, y-axis = normalized amplitude
plt.xlabel("Time (s)")
plt.ylabel("Normalised amplitude")
plt.title("Speech waveform (time domain)")
plt.xlim(0, duration)
plt.ylim(-1.05, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()