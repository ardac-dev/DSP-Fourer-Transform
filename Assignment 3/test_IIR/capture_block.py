import time
import pyfirmata2
import numpy as np
import matplotlib.pyplot as plt

board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)
board.samplingOn(1)   # 1 kHz

samples = []

def a0_callback(data):
    if data is None:
        return
    adc = int(data * 1023)
    samples.append(adc)

a0 = board.get_pin('a:0:i')
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Capturing 2000 samples (~2 seconds). Move in front of the sensor!")
while len(samples) < 2000:
    time.sleep(0.01)

board.exit()
print("Done, plotting...")

arr = np.array(samples)

# subtract DC offset
arr_centered = arr - np.mean(arr)

plt.figure()
plt.plot(arr_centered)
plt.title("Centered A0 signal")
plt.xlabel("Sample index")
plt.ylabel("ADC counts (zero-mean)")
plt.show()
