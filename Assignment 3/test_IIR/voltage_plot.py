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
voltages = deque([0.0]*MAX_SAMPLES, maxlen=MAX_SAMPLES)
times     = deque([0.0]*MAX_SAMPLES, maxlen=MAX_SAMPLES)

start_time = time.time()

# ============================================================
# 3) CALLBACK: CALLED AT EXACT 1 kHz WHEN NEW DATA ARRIVES
# ============================================================
def a0_callback(data):
    if data is None:
        return
    
    voltage = data * 5.0                 # convert 0.0–1.0 to 0–5 V
    t = time.time() - start_time         # timestamp

    voltages.append(voltage)
    times.append(t)

    # Optional: print values (comment out if too spammy)
    # print(f"{t:.6f}, {voltage:.4f}")

# Activate A0
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Sampling at 1 kHz and plotting in real time. CTRL+C to stop.")

# ============================================================
# 4) REAL-TIME PLOTTING SETUP
# ============================================================
plt.style.use('ggplot')
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1.5)

ax.set_title("Real-Time Voltage from A0 (pyFirmata2, 1 kHz)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_ylim(0, 5)          # adjust if you use a different bias
ax.set_xlim(0, 2)          # viewing window ~2 seconds

# Animation function: updates plot every ~50 ms
def update_plot(frame):
    if len(times) > 1:
        ax.set_xlim(times[0], times[-1])
        line.set_data(times, voltages)
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
