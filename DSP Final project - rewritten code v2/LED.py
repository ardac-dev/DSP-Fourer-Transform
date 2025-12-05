class MotionLedController:

    def __init__(self, led_pin, th_high: float, th_low: float, min_toggle_interval_samples: int = 0):
       
        self.led = led_pin
        self.th_high = th_high
        self.th_low = th_low
        self.min_toggle_interval = min_toggle_interval_samples

        self.motion_state = 0  # 0 = IDLE, 1 = ACTIVE
        self.led_state = 0
        self.last_toggle_sample = 0

        if self.led is not None:
            self.led.write(0)

    def update(self, y_clean: float, sample_index: int) -> int:
        
        val = y_clean

        if self.motion_state == 0:
            # rising through threshold
            if val > self.th_high:
                if (sample_index - self.last_toggle_sample) >= self.min_toggle_interval:
                    self.motion_state = 1
                    self.led_state = 1 - self.led_state
                    if self.led is not None:
                        self.led.write(self.led_state)
                    self.last_toggle_sample = sample_index
        else:
            # falling back below low threshold
            if val < self.th_low:
                self.motion_state = 0

        return self.led_state

    def turn_off(self):
        self.led_state = 0
        if self.led is not None:
            self.led.write(0)
