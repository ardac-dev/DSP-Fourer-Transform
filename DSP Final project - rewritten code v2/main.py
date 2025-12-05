import pyfirmata2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from Radar import RadarDSP
from LED import MotionLedController

#application parameters
FS = 1000 # sampling rate in hz
HPF_CUTOFF = 40 # high pass cutoff freq
LPF_CUTOFF = 300 # low pass cutoff freq

PLOT_WINDOW_S = 5.0 # seconds of data shown in plot

# envelope / motion detection for LED
ENV_ALPHA = 0.05 # envelope smoothing factor
TH_HIGH = 0.25 # motion start threshold
TH_LOW = 0.15 # motion end threshold
MIN_LED_TOGGLE_INTERVAL_SAMPLES = int(1.0 * FS)  # e.g. 1 second at 1 kHz

FS_REPORT_INTERVAL_S = 1.0 #printing measured sampling rate every second

# aurdino setup
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1) # request 1 kHz sampling from pyFirmata2

# LED on digital pin D2
led_pin = board.get_pin('d:2:o')


# creating the radar and led controller objects, radar object handles IIR filter and plotting buffers
dsp = RadarDSP(
    fs=FS,
    hpf_cutoff=HPF_CUTOFF,
    lpf_cutoff=LPF_CUTOFF,
    plot_window_s=PLOT_WINDOW_S
)

# led controller
motion_led = MotionLedController(
    led_pin=led_pin,
    env_alpha=ENV_ALPHA,
    th_high=TH_HIGH,
    th_low=TH_LOW,
    min_toggle_interval_samples=MIN_LED_TOGGLE_INTERVAL_SAMPLES
)

# sampling rate monitor state
last_fs_report_time = time.time()
last_fs_report_sample = 0

def a0_callback(data):
    global last_fs_report_time, last_fs_report_sample

    if data is None:
        return

    # firmata gives 0..1 , convert to 0..5V
    voltage = data * 5.0

    # process through cascaded IIR filters
    y = dsp.process_sample(voltage)

    # update LED logic based on filtered signal
    motion_led.update(y, dsp.current_sample)
    
    #checking sampling rate
    now = time.time()
    dt = now - last_fs_report_time

    if dt >= FS_REPORT_INTERVAL_S:
        # how many samples since last report
        dsamples = dsp.current_sample - last_fs_report_sample

        if dt > 0:
            measured_fs = dsamples / dt
            print(f"Measured sampling rate: {measured_fs:.1f} Hz (expected {FS} Hz)")

        # reset for next interval
        last_fs_report_time = now
        last_fs_report_sample = dsp.current_sample

a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Running real time radar DSP at 1 kHz.")

#plotting
plt.style.use('ggplot')

fig, (ax_raw, ax_filt) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

line_raw, = ax_raw.plot([], [], lw=1.0)
ax_raw.set_title("Radar Raw Output (Real-Time)")
ax_raw.set_ylabel("Amplitude (V)")
ax_raw.set_ylim(0, 5)

# filtered signal line
line_filt, = ax_filt.plot([], [], lw=1.2)
ax_filt.set_title("Radar Filtered Output (Real-Time)")
ax_filt.set_xlabel("Time (s)")
ax_filt.set_ylabel("Amplitude (V)")
ax_filt.set_ylim(-2, 3)
ax_filt.set_xlim(0, PLOT_WINDOW_S)

def update_plot(frame):
    
    if len(dsp.times) > 1:
        ax_filt.set_xlim(dsp.times[0], dsp.times[-1])
        line_raw.set_data(dsp.times, dsp.raw_values)
        line_filt.set_data(dsp.times, dsp.filtered_values)
    return line_raw, line_filt

ani = animation.FuncAnimation(fig, update_plot, interval=150, blit=True)

try:
    plt.show()
except KeyboardInterrupt:
    pass

print("Exiting…")
motion_led.turn_off()
board.exit()
