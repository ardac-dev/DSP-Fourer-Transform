import time
import pyfirmata2
import numpy as np

from lpf_IIR import IIRlpf
from hpf_IIR import IIRhpf

# ============================================================
# PARAMETERS
# ============================================================
FS = 1000.0
DURATION = 5.0
N_SAMPLES = int(FS * DURATION)

IGNORE_INITIAL_SECS = 2.0
IGNORE_SAMPLES = int(IGNORE_INITIAL_SECS * FS)

HPF_CUTOFF = 70
LPF_CUTOFF = 300.0

# ============================================================
# SET UP FILTERS (3× HPF + 3× LPF)
# ============================================================
hpf_sections = [IIRhpf(HPF_CUTOFF, FS) for _ in range(3)]
lpf_sections = [IIRlpf(LPF_CUTOFF, FS) for _ in range(3)]

for sec in hpf_sections:
    sec.calc_coeffs()
for sec in lpf_sections:
    sec.calc_coeffs()

def bandpass_filter_sample(x):
    """Apply HPF×3 then LPF×3."""
    y = x
    for sec in hpf_sections:
        y = sec.dofilter(y)
    for sec in lpf_sections:
        y = sec.dofilter(y)
    return y

# ============================================================
# CONNECT TO ARDUINO
# ============================================================
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)  # 1 kHz

raw_samples = []
filtered_samples = []

def a0_callback(data):
    if data is None:
        return
    if len(raw_samples) >= N_SAMPLES:
        return

    v = data * 5.0              # raw ADC in volts
    y = bandpass_filter_sample(v)

    raw_samples.append(v)
    filtered_samples.append(y)

# register callback
a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print(f"\nCollecting {DURATION} seconds of data...")

while len(raw_samples) < N_SAMPLES:
    time.sleep(0.01)

print("Finished capture.\n")
a0.disable_reporting()
board.exit()

# ============================================================
# ANALYZE ONLY THE 2–5 SECOND WINDOW
# ============================================================
raw = np.array(raw_samples)
filt = np.array(filtered_samples)

raw_seg  = raw[IGNORE_SAMPLES:]   # raw from 2–5 seconds
filt_seg = filt[IGNORE_SAMPLES:]  # filtered from 2–5 seconds

# Peak values
max_raw  = np.max(np.abs(raw_seg))
max_filt = np.max(np.abs(filt_seg))

# Min and max (for range)
raw_min, raw_max   = np.min(raw_seg),  np.max(raw_seg)
filt_min, filt_max = np.min(filt_seg), np.max(filt_seg)

raw_range  = raw_max  - raw_min
filt_range = filt_max - filt_min

# ============================================================
# PRINT RESULTS
# ============================================================
print("==============================================")
print("           SIGNAL METRICS (2–5 seconds)")
print("==============================================")
print(f"Max absolute RAW value:       {max_raw:.6f} V")
print(f"Max absolute FILTERED value:  {max_filt:.6f} V")
print("----------------------------------------------")
print(f"RAW signal range (max-min):   {raw_range:.6f} V")
print(f"FILTERED signal range:        {filt_range:.6f} V")
print("==============================================\n")
