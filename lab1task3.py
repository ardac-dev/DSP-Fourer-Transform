# simple_dtmf_threshold_decode.py
import numpy as np
from scipy.io import wavfile
import numpy as np

# DTMF freq tables
DTMF_ROWS = [697, 770, 852, 941]
DTMF_COLS = [1209, 1336, 1477, 1633]
DTMF_MAP = {
    (697,1209):'1', (697,1336):'2', (697,1477):'3', (697,1633):'A',
    (770,1209):'4', (770,1336):'5', (770,1477):'6', (770,1633):'B',
    (852,1209):'7', (852,1336):'8', (852,1477):'9', (852,1633):'C',
    (941,1209):'*', (941,1336):'0', (941,1477):'#', (941,1633):'D'
}

def to_mono_float(x):
    if x.ndim > 1:
        x = x[:,0]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)
    return x

def simple_segment_by_threshold(x, thr=0.5, min_len_s=0.06, min_gap_s=0.08, sr=8000):
    """
    Very simple segmentation:
      - consider |x| > thr as "active"
      - merge small gaps
      - drop short segments
    Returns sample index pairs [(start,end), ...]
    """
    env = np.abs(x)
    active = env > thr

    # collect runs of True
    runs = []
    i = 0
    L = len(active)
    while i < L:
        if active[i]:
            j = i
            while j < L and active[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1

    # merge gaps shorter than min_gap_s
    merged = []
    max_gap = int(min_gap_s * sr)
    for r in runs:
        if not merged:
            merged.append(r)
        else:
            if r[0] - merged[-1][1] <= max_gap:
                merged[-1][1] = r[1]
            else:
                merged.append(r)

    # drop segments shorter than min_len_s
    out = []
    min_len = int(min_len_s * sr)
    for s,e in merged:
        if (e - s) >= min_len:
            out.append((s,e))
    return out

def band_power_fft(x, sr, f0, bw=30.0):
    """
    Power around target frequency f0 by FFT:
      - take middle 60% of the segment to avoid edges
      - rectangular band ±bw Hz around f0, sum magnitude^2
    """
    N = len(x)
    if N < int(0.04*sr):
        return 0.0
    s = int(0.2*N); e = int(0.8*N)
    xm = x[s:e] * np.hanning(e - s)
    X = np.fft.rfft(xm)
    freqs = np.fft.rfftfreq(len(xm), 1/sr)
    band = (freqs >= (f0 - bw)) & (freqs <= (f0 + bw))
    P = np.sum(np.abs(X[band])**2)
    return float(P)

def detect_digit_in_segment(seg, sr):
    # measure power near each row and col frequency
    row_powers = [band_power_fft(seg, sr, f0, bw=25.0) for f0 in DTMF_ROWS]
    col_powers = [band_power_fft(seg, sr, f0, bw=25.0) for f0 in DTMF_COLS]

    # pick strongest in each group
    r_idx = int(np.argmax(row_powers)); r_freq = DTMF_ROWS[r_idx]
    c_idx = int(np.argmax(col_powers)); c_freq = DTMF_COLS[c_idx]

    # simple dominance check (avoid random noise)
    def ok(ps):
        top = max(ps)
        ps2 = sorted(ps)[-2] if len(ps) >= 2 else 0.0
        return (top > 1e-8) and (top / (ps2 + 1e-18) > 2.0)  # ~6 dB

    if ok(row_powers) and ok(col_powers):
        return DTMF_MAP.get((r_freq, c_freq))
    return None

if __name__ == "__main__":
    wav_path = "3.wav"  # <-- set your file
    sr, data = wavfile.read(wav_path)
    x = to_mono_float(data)

    # ---- 1) segment by simple absolute-threshold ----
    # You wanted "if signal > 0.5, it's a press"
    # You can tweak thr/min_len/min_gap to fit your recording
    segments = simple_segment_by_threshold(
        x, thr=0.20, min_len_s=0.06, min_gap_s=0.08, sr=sr
    )

    # ---- 2) decode each segment into a single digit ----
    digits = []
    for i, (s,e) in enumerate(segments):
        seg = x[s:e]
        d = detect_digit_in_segment(seg, sr)
        # "put it into the array in order"
        digits.append(d if d is not None else "?")
        print(f"[{i:02d}] {s}-{e} ({(e-s)/sr*1000:.1f} ms) -> {digits[-1]}")

    number = "".join(d for d in digits if d is not None)
    print("\nDetected digits (raw):", digits)
    print("Detected number:", number)
