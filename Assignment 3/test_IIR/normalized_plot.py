import time
import pyfirmata2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ============================================================
# 1) CONNECT TO ARDUINO USING pyfirmata2
# ============================================================
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)     # 1 kHz sampling

# ============================================================
# 2) DATA BUFFERS (STORE THE LAST N SAMPLES FOR PLOTTING)
# ============================================================
MAX_SAMPLES = 2000             # number of points shown on the screen
proc_values = deque([0.0]*MAX_SAMPLES, maxlen=MAX_SAMPLES)
times       = deque([0.0]*MAX_SAMPLES, maxlen=MAX_SAMPLES)

start_time = time.time()

# DC offset estimate (initially "unknown")
dc_est = None
# how fast DC estimate adapts (small alpha = slow, stable)
DC_ALPHA = 0.001

# digital gain after DC removal (purely software)
DIGITAL_GAIN = 10.0   # try 10, 20, etc.

# ============================================================
# 3) CALLBACK: CALLED AT EXACT 1 kHz WHEN NEW DATA ARRIVES
# ============================================================
def a0_callback(data):
    global dc_est

    if data is None:
        return
    
    # 1) Convert Firmata's normalised value (0..1) to volts (0..5)
    voltage = data * 5.0

    # 2) Timestamp
    t = time.time() - start_time

    # 3) DC offset estimation (running average)
    if dc_est is None:
        dc_est = voltage
    else:
        dc_est = (1.0 - DC_ALPHA) * dc_est + DC_ALPHA * voltage

    # 4) Remove DC offset
    ac = voltage - dc_est     # now centered ~around 0 V

    # 5) Digital amplification / normalisation
    processed = ac * DIGITAL_GAIN

    # 6) Store for plotting
    proc_values.append(processed)
    times.append(t)

    # Optional: print for debugging
    # print(f"{t:.6f}, V={voltage:.4f}, DC={dc_est:.4f}, AC={ac:.5f}, PROC={processed:.5f}")


# Activate A0
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Sampling at 1 kHz, DC-removing + digitally amplifying. CTRL+C to stop.")

# ============================================================
# 4) REAL-TIME PLOTTING SETUP
# ============================================================
plt.style.use('ggplot')
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1.5)

ax.set_title("Microwave sensor (DC removed & digitally amplified)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalised amplitude (arb. units)")
ax.set_ylim(-2, 2)         # adjust depending on DIGITAL_GAIN and signal
ax.set_xlim(0, 2)          # viewing window ~2 seconds

# Animation function: updates plot every ~50 ms
def update_plot(frame):
    if len(times) > 1:
        ax.set_xlim(times[0], times[-1])
        line.set_data(times, proc_values)
    return line,

ani = animation.FuncAnimation(
    fig, update_plot, interval=50, blit=True
)

# ============================================================
# 5) RUN
# ============================================================
try:
    plt.show()
except KeyboardInterrupt:
    pass

print("Exiting…")
board.exit()
