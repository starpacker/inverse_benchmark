from scipy.signal import butter, filtfilt, hilbert


# --- Extracted Dependencies ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
