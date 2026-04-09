from scipy.signal import butter, filtfilt, hilbert


# --- Extracted Dependencies ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to data along the last axis."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y
