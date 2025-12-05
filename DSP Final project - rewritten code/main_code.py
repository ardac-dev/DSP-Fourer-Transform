import time
from IIR2ndOrder import IIR2ndOrder
from IIRChain import IIRChain
from coefficient_generation import butter_hpf_2nd, butter_lpf_2nd
import pyfirmata2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

#=======================================
#Logic overview: 
#PEAK THRESHOLD defines the voltage level that corresponds to a detected peak from the radar output
#everytime we receive a new voltage reading, we store it in a (2*NEIGHBORHOOD + 1)-element queue
#if the middle element is the highest and is above PEAK_THRESHOLD, it is a peak, and we record the sample number
#using the sample rate of 1000 Hz, we calculate the time between consecutive peaks to determine signal frequency
#we use this to calculate the velocity on each new sample

PEAK_THRESHOLD = 0.15
NEIGHBORHOOD = 2

signal_buffer = deque([], maxlen = 2*NEIGHBORHOOD + 1)
last_peak = 0 #sample number of last peak
current_sample = 0 #sample number of current sample

#=======
# ========= ENVELOPE / MOTION DETECTION SETTINGS =========
ENV_ALPHA = 0.05     # smoothing factor (0–1), adjust as needed
TH_HIGH   = 0.25     # upper threshold to detect motion start
TH_LOW    = 0.15     # lower threshold to detect motion end

env_smooth   = 0.0   # smoothed envelope
motion_state = 0     # 0 = IDLE, 1 = ACTIVE

#======

#===================================
#filter initialization
HPF_CUTOFF = 70
LPF_CUTOFF = 200
FS = 1000

lpf_a1,lpf_a2,lpf_b0,lpf_b1,lpf_b2 = butter_lpf_2nd(LPF_CUTOFF, FS)
hpf_a1, hpf_a2, hpf_b0, hpf_b1, hpf_b2 = butter_hpf_2nd(HPF_CUTOFF, FS)

IIR_lpf_1 = IIR2ndOrder(lpf_a1,lpf_a2,lpf_b0,lpf_b1,lpf_b2)
IIR_lpf_2 = IIR2ndOrder(lpf_a1,lpf_a2,lpf_b0,lpf_b1,lpf_b2)
IIR_lpf_3 = IIR2ndOrder(lpf_a1,lpf_a2,lpf_b0,lpf_b1,lpf_b2)

IIR_hpf_1 = IIR2ndOrder(hpf_a1,hpf_a2,hpf_b0,hpf_b1,hpf_b2)
IIR_hpf_2 = IIR2ndOrder(hpf_a1,hpf_a2,hpf_b0,hpf_b1,hpf_b2)
IIR_hpf_3 = IIR2ndOrder(hpf_a1,hpf_a2,hpf_b0,hpf_b1,hpf_b2)

filter_chain = [IIR_lpf_1, IIR_lpf_2, IIR_hpf_1, IIR_hpf_2]

IIR_chain = IIRChain()

for section in filter_chain:
    IIR_chain.addFilter(section)
#====================================
#Creating buffers for plotting data

# Plot buffers (keep window of recent samples)
PLOT_WINDOW_S = 5          # show last 5 seconds

MAX_PLOT_POINTS = PLOT_WINDOW_S * FS

times = deque([], maxlen=MAX_PLOT_POINTS)
filtered_values = deque([], maxlen=MAX_PLOT_POINTS)
raw_values = deque([], maxlen=MAX_PLOT_POINTS)



#=====================================
#Peak Detection and Velocity Calculation Code
fd = 0
v_kmh = 0
def detect_peak():

    #check that first half of buffer is inceasing
    for signal_idx in range(NEIGHBORHOOD):

        if signal_buffer[signal_idx] > signal_buffer[signal_idx + 1]:

            return False
    
    #check that second half of buffer is decreasing
    for signal_idx in range(NEIGHBORHOOD, 2*NEIGHBORHOOD):

        if signal_buffer[signal_idx + 1] > signal_buffer[signal_idx]:

            return False
    
    if signal_buffer[NEIGHBORHOOD] < PEAK_THRESHOLD:

        return False #signal too small, is just noise
    
    return True 

#use doppler formula to calculate velocity (is only called if peak is detected)
def calculate_velocity():

    time_between_peaks = (current_sample - last_peak)/FS

    fd = 1/time_between_peaks

    v_ms = fd * 0.00625 #using doppler formula
    v_kmh = v_ms * 3.6

    return fd, v_kmh

#===============================================================================
#Arduino Setup

board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)
led_state = 0

start_time = time.time()

#===============================================================================
#Callback Function

def a0_callback(data):

    if data is None:
        return
    
    global current_sample, last_peak, fd, v_kmh, led_state, led
    global env_smooth, motion_state
    print(current_sample)
    current_sample += 1

    
    voltage = data * 5

    y = IIR_chain.dofilter(voltage)

    #y = voltage

    signal_buffer.append(y) #add the new filtered value to our signal buffer

    # Need full buffer before checking peaks
    if len(signal_buffer) < 2*NEIGHBORHOOD + 1:
        return

    if detect_peak():

        fd, v_kmh = calculate_velocity()
        last_peak = current_sample
    
    #If it has been more than 30 samples since our last peak (--> we want to reset our velocity to 0)

    if (current_sample - last_peak >= 30):

        fd = 0
        v_kmh = 0
    
    # Update plotting buffers
    t = current_sample / FS            # time in seconds
    times.append(t)
    filtered_values.append(y)
    raw_values.append(voltage)

    '''
    env = abs(y)  # y is the filtered signal

    # Exponential moving average of envelope
    env_smooth = (1.0 - ENV_ALPHA) * env_smooth + ENV_ALPHA * env

    # Simple state machine: IDLE (0) -> ACTIVE (1) -> IDLE
    if motion_state == 0:
        # In IDLE: if we cross high threshold, motion starts
        if env_smooth > TH_HIGH:
            motion_state = 1
            # Rising edge: toggle LED
            led_state = 1 - led_state
            led.write(led_state)
            print(f"MOTION START, LED = {led_state}, env_smooth = {env_smooth:.3f}")
    else:
        # In ACTIVE: if we go below low threshold, motion ends
        if env_smooth < TH_LOW:
            motion_state = 0
            print(f"MOTION END, env_smooth = {env_smooth:.3f}")

    
    '''


    

#================================================================================
#data streaming

# Start streaming
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

led = board.get_pin('d:2:o')

led.write(led_state)


# ============================================================
# plotting filtered and unfiltered values (with velocity displayed)


plt.style.use('ggplot')

fig, (ax_raw, ax_filt) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Raw signal line
line_raw, = ax_raw.plot([], [], lw=1.0)
ax_raw.set_title("Radar Raw Output (Real-Time)")
ax_raw.set_ylabel("Amplitude (V)")
ax_raw.set_ylim(0, 5)          # adjust if needed

# Filtered signal line
line_filt, = ax_filt.plot([], [], lw=1.2)
ax_filt.set_title("Radar Filtered Output (Real-Time)")
ax_filt.set_xlabel("Time (s)")
ax_filt.set_ylabel("Amplitude (V)")
ax_filt.set_ylim(-0.25, 0.25)
ax_filt.set_xlim(0, PLOT_WINDOW_S)

# Velocity display text (on filtered plot)
velocity_text = ax_filt.text(
    0.02, 0.92, "Velocity: 0.00 km/h",
    transform=ax_filt.transAxes,
    fontsize=14,
    color='black',
    bbox=dict(facecolor='white', alpha=0.7)
)

def update_plot(frame):
    if len(times) > 1:
        # Scroll x-axis to follow data
        ax_filt.set_xlim(times[0], times[-1])

        # Update raw and filtered lines
        line_raw.set_data(times, raw_values)
        line_filt.set_data(times, filtered_values)

    # Update velocity text
    velocity_text.set_text(f"Velocity: {v_kmh:5.2f} km/h")

    return line_raw, line_filt, velocity_text

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







