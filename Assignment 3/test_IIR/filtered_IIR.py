import time
import pyfirmata2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

# ============================================================
# 1) CONNECT TO ARDUINO USING pyfirmata2
# ============================================================
FS_HZ = 1000.0   # sampling rate (Hz)
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)     # 1 kHz sampling

# ============================================================
# 2) DATA BUFFERS
# ============================================================
MAX_SAMPLES = 2000   # for plotting (~2s)
filtered_values = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)
times           = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)

# buffer for FFT-based speed estimation
N_FFT = 1024
fft_buffer = deque([0.0] * N_FFT, maxlen=N_FFT)

start_time = time.time()

# ============================================================
# 3) DC OFFSET ESTIMATION
# ============================================================
dc_est = None
DC_ALPHA = 0.001          # how fast DC estimate adapts (small = slow, stable)

DIGITAL_GAIN = 1.0        # extra digital gain after DC removal

# ============================================================
# 3a) BAND-PASS IIR (APPROX 70–650 Hz) - 1st ORDER HP + 1st ORDER LP
#     Discrete-time RC equivalents for Fs = 1000 Hz
#     alpha_HP = tau / (tau + Ts), tau = 1/(2*pi*fc)
#     alpha_LP = Ts / (Ts + tau)
# ============================================================
# High-pass corner ~70 Hz
A_HP = 0.6945298274804017        # ~70 Hz HP

# Low-pass corner ~650 Hz
ALPHA_LP = 0.8033072102559965    # ~650 Hz LP

# Band-pass filter states
prev_x_hp = 0.0    # previous input sample for HP
prev_y_hp = 0.0    # previous HP output
prev_y_lp = 0.0    # previous LP output (band-pass output)

# ============================================================
# 3b) 50 Hz NOTCH FILTER (IIR BIQUAD) TO REMOVE 45–55 Hz
# ============================================================
# Design: Fs = 1000 Hz, f0 = 50 Hz
# Standard notch: b = [1, -2cos(w0), 1], a = [1, -2Rcos(w0), R^2]
# R < 1 controls notch width; R=0.9 => fairly wide (~45–55 Hz)

b0_notch = 1.0
b1_notch = -1.902113032590307   # -2*cos(2*pi*50/1000)
b2_notch = 1.0
a1_notch = -1.7119017293312764  # -2*R*cos(...), R = 0.9
a2_notch = 0.81                 # R^2

# Notch filter states
notch_x1 = 0.0
notch_x2 = 0.0
notch_y1 = 0.0
notch_y2 = 0.0

# ============================================================
# 4) CALLBACK: CALLED AT 1 kHz WHEN NEW DATA ARRIVES
# ============================================================
def a0_callback(data):
    global dc_est, prev_x_hp, prev_y_hp, prev_y_lp
    global notch_x1, notch_x2, notch_y1, notch_y2

    if data is None:
        return
    
    # 1) Convert to volts
    voltage = data * 5.0

    # 2) Time
    t = time.time() - start_time

    # 3) DC offset estimate (IIR low-pass)
    if dc_est is None:
        dc_est = voltage
    else:
        dc_est = (1.0 - DC_ALPHA) * dc_est + DC_ALPHA * voltage

    # 4) Remove DC
    x_ac = voltage - dc_est

    # 5) Digital gain
    x = x_ac * DIGITAL_GAIN

    # ========= IIR BAND-PASS (70–650 Hz) =========
    # High-pass stage
    y_hp = A_HP * (prev_y_hp + x - prev_x_hp)

    # Low-pass stage on HP output -> band-passed
    y_bp = prev_y_lp + ALPHA_LP * (y_hp - prev_y_lp)

    # Update band-pass states
    prev_x_hp = x
    prev_y_hp = y_hp
    prev_y_lp = y_bp

    # ========= 50 Hz IIR NOTCH (removes ~45–55 Hz) =========
    # Direct Form I:
    # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    x_n = y_bp
    y_n = (
        b0_notch * x_n +
        b1_notch * notch_x1 +
        b2_notch * notch_x2 -
        a1_notch * notch_y1 -
        a2_notch * notch_y2
    )

    # Update notch states
    notch_x2 = notch_x1
    notch_x1 = x_n
    notch_y2 = notch_y1
    notch_y1 = y_n

    # 7) Store final (band-pass + notch) result
    filtered_values.append(y_n)
    times.append(t)
    fft_buffer.append(y_n)


# Activate A0
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Sampling at 1 kHz, BP(70–650 Hz) + 50 Hz notch + estimating speed. CTRL+C to stop.")

# ============================================================
# 5) SPEED ESTIMATION VIA FFT
# ============================================================
def estimate_speed_from_fft():
    """Estimate Doppler frequency and speed from the last N_FFT samples."""
    if len(fft_buffer) < N_FFT:
        return None, None, None

    x = np.array(fft_buffer)

    # remove residual DC (just in case)
    x = x - np.mean(x)

    # apply a window to reduce spectral leakage
    w = np.hanning(N_FFT)
    xw = x * w

    # real FFT
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_HZ)
    mag = np.abs(X)

    # ignore very low freq (e.g. below 5 Hz)
    f_min = 5.0
    start_bin = int(f_min * N_FFT / FS_HZ)
    if start_bin < 1:
        start_bin = 1

    peak_idx = np.argmax(mag[start_bin:]) + start_bin
    fd = freqs[peak_idx]  # Doppler frequency in Hz

    # Convert Doppler frequency to speed (for 24 GHz radar)
    # lambda = c / f ≈ 0.0125 m, v = fd * lambda / 2 ≈ fd * 0.00625 m/s
    v_ms  = fd * 0.00625
    v_kmh = v_ms * 3.6

    return fd, v_ms, v_kmh

# ============================================================
# 6) REAL-TIME PLOTTING
# ============================================================
plt.style.use('ggplot')
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1.5)

ax.set_title("Band-pass (70–650 Hz) + 50 Hz notch")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Filtered amplitude (arb. units)")
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(0, 2)

def update_plot(frame):
    if len(times) > 1:
        ax.set_xlim(times[0], times[-1])
        line.set_data(times, filtered_values)

        # Update speed estimation & PRINT in terminal
        fd, v_ms, v_kmh = estimate_speed_from_fft()
        if fd is not None:
            print(f"fd = {fd:6.1f} Hz | v = {v_ms:5.3f} m/s | {v_kmh:5.2f} km/h")
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
