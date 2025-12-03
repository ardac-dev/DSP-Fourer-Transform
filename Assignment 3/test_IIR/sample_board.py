import time
import pyfirmata2

# 1) Connect to the Arduino (auto-detect port)
board = pyfirmata2.Arduino(pyfirmata2.Arduino.AUTODETECT)

# 2) Start sampling at 1 ms interval (1 kHz)
board.samplingOn(1)   # 1 ms = 1000 Hz

# 3) Callback: called every time a new A0 value arrives
def a0_callback(data):
    if data is None:
        return
    adc = int(data * 1023)     # 0.0..1.0 => 0..1023
    print(adc)

# 4) Set up A0 as analogue input, attach callback, enable reporting
a0 = board.get_pin('a:0:i')    # a = analog, 0 = A0, i = input
a0.register_callback(a0_callback)
a0.enable_reporting()

print("Reading A0 at 1 kHz. Press Ctrl+C to stop.")

try:
    while True:
        # keep program alive; callbacks do the work
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    board.exit()
