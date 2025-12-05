from collections import deque

from IIR2ndOrder import IIR2ndOrder
from IIRChain import IIRChain
from coefficient_generation import butter_hpf_2nd, butter_lpf_2nd


class RadarDSP:

    def __init__(self, fs: float, hpf_cutoff: float, lpf_cutoff: float, plot_window_s: float):
        
        self.FS = fs

        # Sample counter
        self.current_sample = 0

        # plotting buffers (raw and filtered signals)
        self.PLOT_WINDOW_S = plot_window_s
        max_points = int(self.PLOT_WINDOW_S * self.FS)

        self.times = deque([], maxlen=max_points)
        self.raw_values = deque([], maxlen=max_points)
        self.filtered_values = deque([], maxlen=max_points)

        self._build_filter_chain(hpf_cutoff, lpf_cutoff)

    def _build_filter_chain(self, hpf_cutoff: float, lpf_cutoff: float) -> None:

        # design 2nd order butterworth low pass and high pass filters sections
        lpf_a1, lpf_a2, lpf_b0, lpf_b1, lpf_b2 = butter_lpf_2nd(lpf_cutoff, self.FS)
        hpf_a1, hpf_a2, hpf_b0, hpf_b1, hpf_b2 = butter_hpf_2nd(hpf_cutoff, self.FS)

        # create multiple sections for a steeper filter
        lpf_1 = IIR2ndOrder(lpf_a1, lpf_a2, lpf_b0, lpf_b1, lpf_b2)
        lpf_2 = IIR2ndOrder(lpf_a1, lpf_a2, lpf_b0, lpf_b1, lpf_b2)
        lpf_3 = IIR2ndOrder(lpf_a1, lpf_a2, lpf_b0, lpf_b1, lpf_b2)

        hpf_1 = IIR2ndOrder(hpf_a1, hpf_a2, hpf_b0, hpf_b1, hpf_b2)
        hpf_2 = IIR2ndOrder(hpf_a1, hpf_a2, hpf_b0, hpf_b1, hpf_b2)
        hpf_3 = IIR2ndOrder(hpf_a1, hpf_a2, hpf_b0, hpf_b1, hpf_b2)

        self.filter_chain = IIRChain()
        for section in [lpf_1, lpf_2, lpf_3, hpf_1, hpf_2, hpf_3]:
            self.filter_chain.addFilter(section)


    def process_sample(self, voltage: float) -> float:

        self.current_sample += 1

        # apply the IIR bandpass filter
        y = self.filter_chain.dofilter(voltage)

        # update plotting buffers
        t = self.current_sample / self.FS
        self.times.append(t)
        self.raw_values.append(voltage)
        self.filtered_values.append(y)

        return y
