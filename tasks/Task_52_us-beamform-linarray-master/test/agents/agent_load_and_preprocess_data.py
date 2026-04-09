import logging

import numpy as np

from scipy import signal

import h5py

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def get_tgc(alpha0, prop_dist, transmit_freq):
    """Time-gain compensation"""
    n = 1  # approx. 1 for soft tissue
    alpha = alpha0 * (transmit_freq * 1e-6)**n
    # tgc_gain = 10^(alpha * dist_cm / 20 * ???) 
    # Original formula: tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    return tgc_gain

def load_and_preprocess_data(data_path, params):
    """
    Loads HDF5 data and performs preprocessing (TGC, Filtering, Apodization).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info("Loading data...")
    with h5py.File(data_path, 'r') as h5f:
        sensor_data = h5f['dataset_1'][:] # Shape: (N_beams, N_channels, Record_length)

    logger.info(f"Raw data shape: {sensor_data.shape}")

    # Extract params
    n_transmit_beams = params['N_TRANSMIT_BEAMS']
    n_probe_channels = params['N_PROBE_CHANNELS']
    transmit_freq = params['TRANSMIT_FREQ']
    speed_sound = params['SPEED_SOUND']
    array_pitch = params['ARRAY_PITCH']
    sample_rate = params['SAMPLE_RATE']
    time_offset = params['TIME_OFFSET']
    
    record_length = sensor_data.shape[2]
    
    # Time vector
    t = np.arange(record_length) / sample_rate - time_offset
    
    # Probe coordinates
    xd = np.arange(n_probe_channels) * array_pitch
    xd = xd - np.max(xd) / 2

    # --- Preprocessing Logic ---
    fs = 1 / (t[1] - t[0])
    a0 = 0.4 # attenuation coeff

    # TGC
    zd = t * speed_sound / 2
    zd2 = zd**2
    dist1 = zd
    tgc = np.zeros((n_probe_channels, record_length))
    for r in range(n_probe_channels):
        dist2 = np.sqrt(xd[r]**2 + zd2)
        prop_dist = dist1 + dist2
        tgc[r, :] = get_tgc(a0, prop_dist, transmit_freq)

    data_amp = np.zeros(sensor_data.shape)
    for m in range(n_transmit_beams):
        data_amp[m, :, :] = sensor_data[m, :, :] * tgc

    # Filtering
    filt_ord = 201
    lc, hc = 0.5e6, 2.5e6
    lc = lc / (fs / 2)
    hc = hc / (fs / 2)
    B_filt = signal.firwin(filt_ord, [lc, hc], pass_zero=False)

    # Interpolation factors
    interp_fact = 4
    fs_new = fs * interp_fact
    record_length2 = record_length * interp_fact
    
    # Apodization Window
    # Fallback logic for Tukey
    try:
        apod_win = signal.tukey(n_probe_channels)
    except AttributeError:
        try:
            from scipy.signal.windows import tukey
            apod_win = tukey(n_probe_channels)
        except ImportError:
            apod_win = np.hanning(n_probe_channels)

    data_apod = np.zeros((n_transmit_beams, n_probe_channels, record_length2))
    
    # Apply Filter, Resample, Apodize
    for m in range(n_transmit_beams):
        for n in range(n_probe_channels):
            w = data_amp[m, n, :]
            data_filt = signal.lfilter(B_filt, 1, w)
            data_interp = signal.resample_poly(data_filt, interp_fact, 1)
            data_apod[m, n, :] = apod_win[n] * data_interp

    # New time vector correction
    freqs, delay_grp = signal.group_delay((B_filt, 1))
    delay = int(delay_grp[0]) * interp_fact
    t2 = np.arange(record_length2) / fs_new + t[0] - delay / fs_new

    # Remove signal before t=0
    f = np.where(t2 < 0)[0]
    if len(f) > 0:
        cut_idx = f[-1] + 1
        t2 = np.delete(t2, f)
        data_apod = data_apod[:, :, cut_idx:]
    
    # Coordinates of Transmit Beams
    xd_beams = np.arange(n_transmit_beams) * array_pitch
    xd_beams = xd_beams - np.max(xd_beams) / 2

    return data_apod, t2, xd, xd_beams
