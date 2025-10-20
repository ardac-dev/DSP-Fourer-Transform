from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

#touch tone detection DTMF constants
DTMF_ROWS = np.array([697, 770, 852, 941], dtype=float)
DTMF_COLS = np.array([1209, 1336, 1477, 1633], dtype=float)
DTMF_MAP = {
    (697,1209):'1', (697,1336):'2', (697,1477):'3', (697,1633):'A',
    (770,1209):'4', (770,1336):'5', (770,1477):'6', (770,1633):'B',
    (852,1209):'7', (852,1336):'8', (852,1477):'9', (852,1633):'C',
    (941,1209):'*', (941,1336):'0', (941,1477):'#', (941,1633):'D'
}

#this function shows which sections of the wav file have button press 
# and returns an array containing tuples of start and end indexes of the numpy array

def button_press_segment_detector(x, start_threshold = 0.8, sampling_length = 400, noise_threshold = 0.05):
    segments = []
    abs_x = np.abs(x)
    num_peaks = 0
    for i in range(len(abs_x)):
        if abs_x[i] >= start_threshold:
            num_peaks += 1
            if (np.mean(abs_x[i:i+sampling_length]) >= noise_threshold):
                #print(i)
                if (np.max(abs_x[i+1:i+100]) < start_threshold):
                    start = i
                    end = i + sampling_length
                    segments.append((start, end))
    return segments

#this function analyses each detected button press segment using FFT, 
# identifies the two dominant DTMF frequencies (one low, one high) 
# matches them to the DTMF frequency table, and returns the detected digits.
def detect_which_button_is_pressed(segments, sample_rate, x):
    digits = []
    for (start, end) in segments:
        segment = x[start:end]
        X = np.fft.fft(segment)
        num_samples = len(segment)
        k = np.arange(num_samples)
        freqs = np.where(k < num_samples/2, k * sample_rate / num_samples, (k - num_samples) * sample_rate / num_samples)
        magnitudes = np.abs(X)
        
        #only keeping positive magnitudes and frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitudes = magnitudes[pos_mask]
        
        #creating boolean values to seperate the low frequencies (rows from DTMF table 600-1000Hz) 
        # and high frequencies(columns from DTMF table 1150-1700Hz)
        row_band = (freqs >= 600) & (freqs <= 1000)
        col_band = (freqs >= 1150) & (freqs <= 1700)

        f_row = freqs[row_band][np.argmax(magnitudes[row_band])] #finds the strongest frequency in the low frequency row range
        f_col = freqs[col_band][np.argmax(magnitudes[col_band])] #finds the strongest frequency in the high frequency column range

        # matching the detected frequencies both in column and row to the closest DTMF frequencies by finding the smallest absolute difference
        row = DTMF_ROWS[np.argmin(np.abs(DTMF_ROWS - f_row))]
        col = DTMF_COLS[np.argmin(np.abs(DTMF_COLS - f_col))]

        #look at the DTMF map and get the corresponding key and then append it to the array of detected digits
        key = DTMF_MAP.get((row, col), '?')
        digits.append(key)

    return digits

if __name__ == "__main__":

    sample_rate, data = wavfile.read("2.wav")
    td_normalized = (data.astype(np.float32)) / np.max(data)

    segments = button_press_segment_detector(td_normalized)
    digits = detect_which_button_is_pressed(segments, sample_rate, td_normalized)
    print("The number in 2.wav is", ''.join(digits))

    sample_rate, data = wavfile.read("3.wav")
    td_normalized = (data.astype(np.float32)) / np.max(data)

    segments = button_press_segment_detector(td_normalized)
    digits = detect_which_button_is_pressed(segments, sample_rate, td_normalized)
    print("The number in 3.wav is", ''.join(digits))