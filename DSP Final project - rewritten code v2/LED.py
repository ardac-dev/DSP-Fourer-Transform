class MotionLedController:

    def __init__(self, led_pin, env_alpha: float, th_high: float, th_low: float, min_toggle_interval_samples: int = 0):
       
        self.led = led_pin
        self.env_alpha = env_alpha #smoothing factor for the envelope
        self.th_high = th_high
        self.th_low = th_low
        self.min_toggle_interval = min_toggle_interval_samples #using this to prevent very fast blinking

        # internal states
        self.env_smooth = 0.0 # smoothed envelope
        self.motion_state = 0 # 0 = IDLE, 1 = ACTIVE
        self.led_state = 0 # 0 = OFF, 1 = ON
        self.last_toggle_sample = 0 # sample index of last LED toggle

        # ensure led starts in off state
        if self.led is not None:
            self.led.write(0)

    def update(self, y_clean: float, sample_index: int) -> int:
        
        # envelope of the filtered signal
        env = abs(y_clean)

        # exponential moving average for smooth envelope
        self.env_smooth = (1.0 - self.env_alpha) * self.env_smooth \
                          + self.env_alpha * env

        if self.motion_state == 0:
            # in idle check if motion starts
            if self.env_smooth > self.th_high:
                # enforce minimum distance between toggles
                if (sample_index - self.last_toggle_sample) >= self.min_toggle_interval:
                    self.motion_state = 1
                    # toggle led on motion start
                    self.led_state = 1 - self.led_state
                    if self.led is not None:
                        self.led.write(self.led_state)
                    self.last_toggle_sample = sample_index
        else:
            # wait until envelope drops below low threshold
            if self.env_smooth < self.th_low:
                self.motion_state = 0

        return self.led_state

    def turn_off(self) -> None:
        #turn the led off and reset internal led state.
        self.led_state = 0
        if self.led is not None:
            self.led.write(0)
