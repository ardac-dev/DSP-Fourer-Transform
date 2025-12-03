import time
from lpf_IIR import IIRlpf
from hpf_IIR import IIRhpf
import pyfirmata2
from collections import deque
import numpy as np
import tkinter as tk

# ============================================================
# PEAK DETECTION SETTINGS
# ============================================================
PEAK_THRESHOLD = 0.013
NEIGHBORHOOD = 2
MAX_PEAK_HISTORY = 20

signal_buffer = deque([0.0]*100, maxlen=100)
peak_times = deque([], maxlen=MAX_PEAK_HISTORY)
current_velocity_kmh = 0.0

# ============================================================
# FILTER SETTINGS
# ============================================================
FS = 1000.0
HPF_CUTOFF = 70
LPF_CUTOFF = 300

hpf_sections = [IIRhpf(HPF_CUTOFF, FS) for _ in range(3)]
lpf_sections = [IIRlpf(LPF_CUTOFF, FS) for _ in range(3)]

for sec in hpf_sections:
    sec.calc_coeffs()
for sec in lpf_sections:
    sec.calc_coeffs()

# ============================================================
# FILTER PIPELINE
# ============================================================
def bandpass_filter_sample(x):
    y = x
    for sec in hpf_sections:
        y = sec.dofilter(y)
    for sec in lpf_sections:
        y = sec.dofilter(y)
    return y

# ============================================================
# PEAK DETECTION
# ============================================================
def detect_peak(sample, current_time):
    signal_buffer.append(sample)

    if len(signal_buffer) < 1 + 2*NEIGHBORHOOD:
        return False

    idx = len(signal_buffer) - 1 - NEIGHBORHOOD
    center = signal_buffer[idx]

    if center < PEAK_THRESHOLD:
        return False

    for i in range(1, NEIGHBORHOOD + 1):
        if signal_buffer[idx - i] >= center:
            return False
        if signal_buffer[idx + i] >= center:
            return False

    peak_times.append(current_time)
    return True

def time_since_last_peak(current_time):
    if len(peak_times) == 0:
        return float('inf')
    return current_time - peak_times[-1]

def compute_velocity_from_peaks():
    if len(peak_times) < 2:
        return None, None

    intervals = [
        peak_times[i] - peak_times[i-1]
        for i in range(1, len(peak_times))
    ]

    avg_period = sum(intervals) / len(intervals)
    if avg_period <= 0:
        return None, None

    fd = 1.0 / avg_period
    v_ms = fd * 0.00625
    v_kmh = v_ms * 3.6

    return fd, v_kmh

# ============================================================
# ARDUINO SETUP
# ============================================================
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)

start_time = time.time()

# ============================================================
# TKINTER WINDOW (DIGITAL SPEED DISPLAY)
# ============================================================
root = tk.Tk()
root.title("Doppler Speed Display")

# make the window large and simple
root.geometry("800x400")   # adjust as needed
root.configure(bg="black")

velocity_label = tk.Label(
    root,
    text="Velocity: 0.00 km/h",
    font=("Arial", 80, "bold"),
    fg="lime",
    bg="black"
)
velocity_label.pack(expand=True)

def update_gui():
    """Update Tkinter label. Called repeatedly."""
    velocity_label.config(text=f"Velocity: {current_velocity_kmh:5.2f} km/h")
    root.after(50, update_gui)  # update at 20 FPS

root.after(50, update_gui)

# ============================================================
# ARDUINO CALLBACK (REAL-TIME)
# ============================================================
def a0_callback(data):
    global current_velocity_kmh

    if data is None:
        return

    voltage = data * 5.0
    t = time.time() - start_time

    y = bandpass_filter_sample(voltage)
    detect_peak(y, t)

    # If no peaks recently → reset to 0
    if time_since_last_peak(t) > 0.1:
        current_velocity_kmh = 0.0
    else:
        fd, v_kmh = compute_velocity_from_peaks()
        if fd is not None:
            current_velocity_kmh = v_kmh

a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

# ============================================================
# RUN TKINTER LOOP (NO GRAPH)
# ============================================================
try:
    root.mainloop()
except KeyboardInterrupt:
    pass

print("Exiting…")
board.exit()
