import matplotlib

matplotlib.use("Agg")

from scipy.signal import butter, sosfiltfilt, find_peaks

def bandpass_filter(recording, low, high, sampling_rate):
    """Apply bandpass filter to recording."""
    sos = butter(3, [low, high], btype="band", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, recording, axis=0)
