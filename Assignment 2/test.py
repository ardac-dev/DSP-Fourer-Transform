import numpy as np
import matplotlib.pyplot as plt

filename = "mitbih_test.csv"
data = np.loadtxt(filename, delimiter=',')

# --- BÜTÜN MATRİSİ SATIR SATIR TEK SİNYALE DÖNÜŞTÜR ---
signal = data.flatten()

# Örnekleme frekansı
fs = 360

# İlk 10 saniye için kaç örnek?
n = int(10 * fs)

# Eğer sinyal 10 saniyeden kısaysa güvenlik:
n = min(n, len(signal))

# İLK 10 SANİYENİN SİNYALİ
signal_10s = signal[:n]
time_10s = np.arange(n) / fs

# Plot
plt.figure(figsize=(12,4))
plt.plot(time_10s, signal_10s, linewidth=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("ECG (First 10 seconds)")
plt.grid(True)
plt.show()
