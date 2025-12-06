DSP Assignment 3 CDM324 Radar IIR Filtering

1. Project overview:

This project implements real-time IIR filtering of a radar signal using an Arduino Nano and a CDM324 24 GHz Doppler radar module.
The raw IF output of the radar is DC-biased and amplified using an analogue front-end, then sampled by the Arduino at 1 kHz. In Python, the signal is passed through a cascaded chain of 2nd-order Butterworth IIR sections to form a band-pass filter (40–300 Hz). The filtered signal is used to detect hand motion and toggle an LED without physical contact.
The project demonstrates:
- a real-time measurement problem that requires filtering,
- the design and implementation of 2nd-order IIR sections and a filter chain,
- sampling-rate monitoring during real-time operation,
- a physical demo of contactless LED control.

YouTube demo: https://youtu.be/JUmlmh1xke4

2. File structure
- main.py:
Real-time application. Sets up Arduino, runs sampling, calls DSP pipeline, toggles LED, plots signals.
- Radar.py:
RadarDSP class. Builds the band-pass IIR chain and stores raw/filtered samples for plotting.
- coefficient_generation.py:
Generates 2nd-order Butterworth LPF/HPF coefficients (a1, a2, b0, b1, b2).
- IIR2ndOrder.py:
Single 2nd-order IIR filter running sample-by-sample.
- IIRChain.py:
Cascades multiple IIR2ndOrder filters into a higher-order response.
- LED.py:
Motion detection + LED toggle logic.
- test_IIR.py:
Unit tests for both the 2nd-order filter and the cascaded chain.
- report.pdf:
Written report for Assignment 3.

3. Hardware required
- Arduino Nano (or similar)
- CDM324 Doppler radar module
- Analogue front-end:
      - DC-bias network
      - Op-amp amplifier (gain ≈ 50 for AC)
- LED + resistor on digital out pin D2
- Breadboard + jumper wires
- USB connection to PC

Arduino reads the radar output on A0 and drives the LED via D2.

4. Software requirements

- Python 3.x
- pyfirmata2
- numpy
- matplotlib

Install dependencies:

pip install pyfirmata2 numpy matplotlib

The code uses pyfirmata2.Arduino.AUTODETECT, so manual port selection is usually unnecessary.

5. How to run the real-time application

5.1 Connect the hardware
- Power CDM324 from Arduino 5V + GND
- Feed IF output → DC-bias + amplifier
- Amplifier output → Arduino A0
- LED + resistor → D2 → GND
- Connect Arduino to PC via USB

5.2 Run the program
- From the project folder:

python main.py

The program will:
- start sampling at 1 kHz
- apply the cascaded 40–300 Hz IIR band-pass filter
- update the LED according to motion detection
- print measured sampling rate once per second
- open a real-time plot showing:
        - raw radar signal
        - filtered radar signal

5.3 Motion detection behaviour
- Move your hand in front of the radar
- Each detected motion toggles the LED
- If no further motion occurs, LED stays in its last state

5.4 Stopping the program
- Close the plot window or
- Press Ctrl + C

The script turns off the LED and cleanly closes the Arduino connection.

6. How to run the unit tests 
- Run:
python -m unittest test_IIR.py

- The tests verify:
        - identity behaviour with trivial coefficients
        - moving-sum impulse response
        - convolution result of cascaded filters

7. Notes and parameters

Sampling rate
- FS = 1000 Hz
- Real-time sampling rate is continuously measured and printed

Band-pass filter
- High-pass: 40 Hz
- Low-pass: 300 Hz
- Implemented with cascaded 2nd-order Butterworth sections

LED thresholds

Defined in main.py:
- TH_HIGH activates motion
- TH_LOW returns to idle

Thresholds may need retuning depending on signal amplitude, noise and amplifier behaviour.

Optional FFT plot

main.py includes a commented out FFT block used for analysis.
Not required for real time operation.

8. Known limitations
- Breadboard wiring introduces additional noise
- CDM324 output is very small which gives limited range
- Analogue gain must be tuned carefully
- Threshold values depend on hardware setup
- Radar signal strength varies with angle and distance

Despite these issues, the system reliably detects nearby hand motion and shows how cascaded IIR filters can extract weak signals in real time.

