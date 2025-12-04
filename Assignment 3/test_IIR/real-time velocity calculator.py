import time
from lpf_IIR import IIRlpf
from hpf_IIR import IIRhpf
import pyfirmata2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

# ============================================================
# PEAK DETECTION SETTINGS
# ============================================================
PEAK_THRESHOLD = 0.05
NEIGHBORHOOD = 2
MAX_PEAK_HISTORY = 20

signal_buffer = deque([0.0]*100, maxlen=100)
peak_times = deque([], maxlen=MAX_PEAK_HISTORY)

current_velocity_kmh = 0.0   # <- displayed in plot

# ============================================================
# FILTER SETTINGS
# ============================================================
FS = 1000.0
HPF_CUTOFF = 90
LPF_CUTOFF = 300

hpf_sections = [IIRhpf(HPF_CUTOFF, FS) for _ in range(3)]
lpf_sections = [IIRlpf(LPF_CUTOFF, FS) for _ in range(3)]


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

NUM_SAMPLES = 5000
filtered_values = deque([0.0]*NUM_SAMPLES, maxlen=NUM_SAMPLES)
times = deque([0.0]*NUM_SAMPLES, maxlen=NUM_SAMPLES)
start_time = time.time()

# ============================================================
# CALLBACK
# ============================================================
def a0_callback(data):
    global current_velocity_kmh

    if data is None:
        return

    voltage = data * 5.0
    t = time.time() - start_time

    y = bandpass_filter_sample(voltage)

    filtered_values.append(y)
    times.append(t)

    detect_peak(y, t)

    # Real-time velocity every callback
    if time_since_last_peak(t) > 0.1:
        current_velocity_kmh = 0.0
    else:
        fd, v_kmh = compute_velocity_from_peaks()
        if fd is not None:
            current_velocity_kmh = v_kmh

# Start streaming
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

# ============================================================
# PLOT
# ============================================================
plt.style.use('ggplot')
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1.2)

ax.set_title("Radar Filtered Output (Real-Time)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (V)")
ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, 5)

# Velocity display text
velocity_text = ax.text(
    0.02, 0.92, "Velocity: 0.00 km/h",
    transform=ax.transAxes,
    fontsize=14,
    color='black',
    bbox=dict(facecolor='white', alpha=0.7)
)

def update_plot(frame):
    if len(times) > 1:
        ax.set_xlim(times[0], times[-1])
        line.set_data(times, filtered_values)

    # Update velocity text
    velocity_text.set_text(f"Velocity: {current_velocity_kmh:5.2f} km/h")

    return line, velocity_text

ani = animation.FuncAnimation(fig, update_plot, interval=150, blit=True)

# ============================================================
# RUN
# ============================================================
try:
    plt.show()
except KeyboardInterrupt:
    pass

print("Exiting…")
board.exit()


