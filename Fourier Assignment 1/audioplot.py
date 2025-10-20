from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

#Task 1.1 - Reading waveform and plotting normalized amplitude vs time

sample_rate, data = wavfile.read("original_speech.wav")

#normalizing the signal to be in between -1 and 1 range
td_normalized = data.astype(np.float32) / np.iinfo(data.dtype).max

# creating time axis
num_samples = len(td_normalized)
t = np.arange(num_samples) / sample_rate #creating a time array for each sample
duration = num_samples / sample_rate # finding the total duration of the recording

#plotting normalised amplitude vs time
plt.figure(figsize=(10,3))
plt.plot(t, td_normalized, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Normalised amplitude")
plt.title("Speech waveform (time domain)")
plt.xlim(0, duration)
plt.ylim(-1.05, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Task 1.2 - Plotting magnitude (dB) vs Frequency (decades)

fd_mag_raw = np.fft.fft(td_normalized)
k = np.arange(num_samples)
freq_raw = np.where(k < num_samples/2, k * sample_rate / num_samples, (k - num_samples) * sample_rate / num_samples)
pos_mask = freq_raw >= 0
fd_db = 20 * np.log10(2/num_samples*np.abs(fd_mag_raw))[pos_mask]
freq = freq_raw[pos_mask]

plt.plot (freq, fd_db)
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.title('Speech waveform (frequency domain)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Task 1.3 On document

#Task 1.4 Plotting code

plt.plot (freq, fd_db)
plt.axvline(x=2000, color='red', linestyle='--', linewidth=0.8)
plt.axvline(x=8000, color='red', linestyle='--', linewidth=0.8)
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.title('Speech waveform (frequency domain)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Task 1.5 

plt.plot (freq, fd_db)
plt.axvline(x=80, color='red', linestyle='--', linewidth=0.8)
plt.axvline(x=8000, color='red', linestyle='--', linewidth=0.8)
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.title('Speech waveform (frequency domain)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
