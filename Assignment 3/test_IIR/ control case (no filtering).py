import time
import pyfirmata2
from collections import deque
import numpy as np
import tkinter as tk

# ============================================================
# PEAK DETECTION SETTINGS
# ============================================================
PEAK_THRESHOLD = 0.03 + 2.508175
NEIGHBORHOOD = 2
MAX_PEAK_HISTORY = 20

signal_buffer = deque([0.0]*100, maxlen=100)
peak_times = deque([], maxlen=MAX_PEAK_HISTORY)
current_velocity_kmh = 0.0

# ============================================================
# ADC / SAMPLING SETTINGS
# ============================================================
FS = 1000.0
start_time = time.time()
sample_index = 0      # we use sample index instead of time.time() (MUCH more accurate)

# ============================================================
# PEAK DETECTION
# ============================================================
def detect_peak(sample, t):
    signal_buffer.append(sample)

    if len(signal_buffer) < 1 + 2*NEIGHBORHOOD:
        return False

    center_idx = len(signal_buffer) - 1 - NEIGHBORHOOD
    center_val = signal_buffer[center_idx]

    if center_val < PEAK_THRESHOLD:
        return False

    for i in range(1, NEIGHBORHOOD+1):
        if signal_buffer[center_idx - i] >= center_val:
            return False
        if signal_buffer[center_idx + i] >= center_val:
            return False

    peak_times.append(t)
    return True

def time_since_last_peak(t):
    if len(peak_times) == 0:
        return float('inf')
    return t - peak_times[-1]

# ============================================================
# PEAK-BASED VELOCITY ESTIMATION
# ============================================================
def compute_velocity():
    if len(peak_times) < 2:
        return None, None

    intervals = [peak_times[i] - peak_times[i-1] for i in range(1, len(peak_times))]
    avg_period = sum(intervals) / len(intervals)
    if avg_period <= 0:
        return None, None

    fd = 1.0 / avg_period
    v_ms = fd * 0.00625      # for 24 GHz
    v_kmh = v_ms * 3.6

    return fd, v_kmh

# ============================================================
# TKINTER DISPLAY (BIG SPEED)
# ============================================================
root = tk.Tk()
root.title("Doppler Speed Display")
root.geometry("900x500")
root.configure(bg="black")

velocity_label = tk.Label(
    root,
    text="Velocity: 0.00 km/h",
    font=("Arial", 100, "bold"),
    fg="lime",
    bg="black"
)
velocity_label.pack(expand=True)

def update_gui():
    velocity_label.config(text=f"Velocity: {current_velocity_kmh:5.2f} km/h")
    root.after(50, update_gui)

root.after(50, update_gui)

# ============================================================
# ARDUINO SETUP
# ============================================================
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)

# ============================================================
# ARDUINO CALLBACK (1 kHz)
# ============================================================
def a0_callback(data):
    global sample_index, current_velocity_kmh

    if data is None:
        return

    voltage = data * 5.0

    # precise timestamp based on sample index
    sample_index += 1
    t = sample_index / FS

    # peak detection on RAW SIGNAL
    detect_peak(voltage, t)

    # velocity logic
    if time_since_last_peak(t) > 0.1:
        current_velocity_kmh = 0.0
    else:
        fd, v_kmh = compute_velocity()
        if fd is not None:
            current_velocity_kmh = v_kmh

# attach callback
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

# ============================================================
# RUN
# ============================================================
try:
    root.mainloop()
except KeyboardInterrupt:
    pass

print("Exiting…")
board.exit()
