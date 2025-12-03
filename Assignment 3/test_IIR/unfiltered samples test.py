import pyfirmata2
import time
import numpy as np

N_SAMPLES = 5000
samples = []

board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)   # 1 kHz sampling

def a0_callback(data):
    if data is None:
        return

    voltage = data * 5.0   # convert ADC reading to volts
    samples.append(voltage)

    if len(samples) >= N_SAMPLES:
        a0.disable_reporting()

a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Collecting 5000 samples...")

while len(samples) < N_SAMPLES:
    time.sleep(0.001)

print("Done.")

avg = np.mean(samples)
print(f"Average of 5000 unfiltered samples: {avg:.6f} V")

board.exit()
