import time
from lpf_IIR import IIRlpf
from hpf_IIR import IIRhpf
import pyfirmata2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

# Peak detection buffers
PEAK_THRESHOLD = 0.03      # volts
NEIGHBORHOOD = 3           # 3 samples each side
MAX_PEAK_HISTORY = 20      # store last few peaks

signal_buffer = deque([0.0]*100, maxlen=100)  # last 100 filtered samples
peak_times = deque([], maxlen=MAX_PEAK_HISTORY)
LAST_VELOCITY_PRINT = 0.0
FS = 1000.0   # sampling rate (Hz)
HPF_CUTOFF = 70 # HPF cutoff freq (Hz)
LPF_CUTOFF = 300 # LPF cutoff freq (Hz)

my_LPF1 = IIRlpf(LPF_CUTOFF, FS)
my_LPF2 = IIRlpf(LPF_CUTOFF, FS)
my_LPF3 = IIRlpf(LPF_CUTOFF, FS)
my_HPF1 = IIRhpf(HPF_CUTOFF, FS)
my_HPF2 = IIRhpf(HPF_CUTOFF, FS)
my_HPF3 = IIRhpf(HPF_CUTOFF, FS)

board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)     # 1 kHz sampling


NUM_SAMPLES = 5000
filtered_values = deque([0.0] * NUM_SAMPLES, maxlen=NUM_SAMPLES)
times = deque([0.0] * NUM_SAMPLES, maxlen=NUM_SAMPLES)

start_time = time.time()


def a0_callback(data):

    if data is None:
        return
    
    voltage_reading = data * 5.0

    lpf1 = my_LPF1.dofilter(voltage_reading)
    lpf2 = my_LPF2.dofilter(lpf1)
    lpf3 = my_LPF3.dofilter(lpf2)
    hpf1 = my_HPF1.dofilter(lpf3)
    hpf2 = my_HPF2.dofilter(hpf1)
    hpf3 = my_HPF3.dofilter(hpf2)

    filtered_signal = hpf3

    filtered_values.append(filtered_signal)

    times.append(time.time() - start_time)

#telling program to read in a0 whenever it comes in, and to run a0_callback whenever new data comes
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()


plt.style.use('ggplot')
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1.5)

ax.set_title("Radar Output")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Filtered amplitude (arb. units)")
ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, 5)

def update_plot(frame):
    if len(times) > 1:
        ax.set_xlim(times[0], times[-1])
        line.set_data(times, filtered_values)

    return line,

ani = animation.FuncAnimation(
    fig, update_plot, interval=200, blit=True)   # update every 200 ms

# ============================================================
# 7) RUN
# ============================================================
try:
    plt.show()
except KeyboardInterrupt:
    pass

print("Exiting…")
board.exit()









