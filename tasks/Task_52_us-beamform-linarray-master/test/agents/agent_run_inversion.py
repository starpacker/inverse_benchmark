import logging

import numpy as np

from scipy import signal

from scipy.interpolate import RectBivariateSpline

import h5py

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def arange2(start, stop=None, step=1):
    """Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop is None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a

def get_tgc(alpha0, prop_dist, transmit_freq):
    """Time-gain compensation"""
    n = 1  # approx. 1 for soft tissue
    alpha = alpha0 * (transmit_freq * 1e-6)**n
    # tgc_gain = 10^(alpha * dist_cm / 20 * ???) 
    # Original formula: tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    return tgc_gain

def envel_detect(scan_line):
    """Envelope detection using Hilbert transform"""
    envelope = np.abs(signal.hilbert(scan_line))
    return envelope

def log_compress(data, dynamic_range, reject_level, bright_gain):
    """Log compression"""
    # Avoid log(0)
    data_safe = data.copy()
    data_safe[data_safe < 0] = 0
    xd_b = 20 * np.log10(data_safe + 1)
    xd_b2 = xd_b - np.max(xd_b)
    xd_b3 = xd_b2 + dynamic_range
    xd_b3[xd_b3 < 0] = 0
    xd_b3[xd_b3 <= reject_level] = 0
    xd_b3 = xd_b3 + bright_gain
    xd_b3[xd_b3 > dynamic_range] = dynamic_range
    return xd_b3

def scan_convert(data, xb, zb):
    """Scan conversion to image grid"""
    decim_fact = 8
    
    # Decimate input data to save compute during interpolation
    data_dec = data[:, 0:-1:decim_fact]
    zb_dec = zb[0:-1:decim_fact]

    # Original code used interp2d. RectBivariateSpline is the modern robust equivalent for rectilinear grids.
    # Note: RectBivariateSpline takes (x, y) grid axes and z data.
    # Here input axes are xb (lateral), zb (depth). Data is (lateral, depth).
    
    # Ensure sorted order for Spline
    # xb and zb are usually sorted, but let's be safe or just pass them.
    
    interpolator = RectBivariateSpline(xb, zb_dec, data_dec)
    
    dz = zb[1] - zb[0]
    # Define new grid
    xnew = arange2(xb[0], xb[-1] + dz, dz)
    znew = zb
    
    # Evaluate spline
    image_sC = interpolator(xnew, znew)
    
    # RectBivariateSpline returns (xnew, znew). We might need to transpose depending on display convention
    # The original interp2d output shape convention matches (znew, xnew) in the usage `interp_func(znew, xnew)`.
    # Let's align with original output: image_sC was result of interp2d(znew, xnew).
    # interp2d (x, y, z) -> call(new_x, new_y). 
    # Original: interp2d(zb, xb, data). x=zb, y=xb. call(znew, xnew).
    # Result shape was (len(xnew), len(znew)) effectively or transposed?
    # Actually, interp2d builds function f(x, y). 
    # Calling f(znew, xnew) evaluates at grid defined by znew and xnew.
    # This implies the result is on meshgrid(znew, xnew).
    
    # To be safe and strictly follow logic:
    # We want output image where axis 0 is X and axis 1 is Z (or vice versa).
    # Typically ultrasound images are (Depth, Lateral) or (Lateral, Depth).
    # The original code transposed at the very end `image_final = ... .T`.
    
    # Let's just return the grid computed by RectBivariateSpline(xb, zb_dec, data_dec)(xnew, znew)
    # This returns shape (len(xnew), len(znew)).
    
    return image_sC, znew, xnew

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

def forward_operator(data_preproc, time_vec, probe_coords, params, method='dynamic'):
    """
    Performs beamforming (The core operator).
    
    Args:
        data_preproc: (N_beams, N_channels, Time)
        time_vec: Time array corresponding to data
        probe_coords: X coordinates of probe elements
        params: Dictionary of constants
        method: 'dynamic' or 'fixed'
    
    Returns:
        image_lines: (N_beams, Depth_samples) raw RF beamformed data
    """
    speed_sound = params['SPEED_SOUND']
    n_probe_channels = params['N_PROBE_CHANNELS']
    n_transmit_beams = params['N_TRANSMIT_BEAMS']
    
    fs = 1 / (time_vec[1] - time_vec[0])
    
    if method == 'fixed':
        # Fixed Focus Beamforming
        receive_focus = 30e-3 # Fixed focal depth
        
        # Calculate delays for fixed focus
        delay_ind = np.zeros(n_probe_channels, dtype=int)
        for r in range(n_probe_channels):
            delay = receive_focus / speed_sound * (np.sqrt((probe_coords[r] / receive_focus)**2 + 1) - 1)
            delay_ind[r] = int(round(delay * fs))
        max_delay = np.max(delay_ind)

        waveform_length = data_preproc.shape[2]
        image_lines = np.zeros((n_transmit_beams, waveform_length))
        
        for q in range(n_transmit_beams):
            scan_line = np.zeros(waveform_length + max_delay)
            for r in range(n_probe_channels):
                pad_len = delay_ind[r]
                waveform = data_preproc[q, r, :]
                
                # Construct padded array manually to match shapes
                # padding at end to match scan_line length: (waveform_length + max_delay) - waveform_len - pad_len
                total_len = waveform_length + max_delay
                remaining = total_len - len(waveform) - pad_len
                
                # Logic from original code: concat(fill_pad, waveform, delay_pad)
                # Original: fill_pad = zeros(len(scan_line) - waveform_length - delay_ind[r])
                # The original logic shifted signal. 
                # Let's reproduce:
                fill_pad = np.zeros(remaining) 
                delay_pad = np.zeros(pad_len)
                
                # Note: Original code logic: scan_line = scan_line + np.concatenate((fill_pad, waveform, delay_pad))
                # This essentially shifts the waveform in the buffer.
                shifted_waveform = np.concatenate((fill_pad, waveform, delay_pad))
                
                # Safety check for size
                if len(shifted_waveform) != len(scan_line):
                     # Adjust if slight rounding error
                     diff = len(scan_line) - len(shifted_waveform)
                     if diff > 0:
                         shifted_waveform = np.concatenate((shifted_waveform, np.zeros(diff)))
                     elif diff < 0:
                         shifted_waveform = shifted_waveform[:diff]

                scan_line = scan_line + shifted_waveform
            
            image_lines[q, :] = scan_line[max_delay:]
            
    elif method == 'dynamic':
        # Dynamic Focusing Beamforming
        zd = time_vec * speed_sound / 2
        zd2 = zd**2
        
        # Precompute indices
        prop_dist = np.zeros((n_probe_channels, len(zd)))
        for r in range(n_probe_channels):
            dist1 = zd # Receive distance (approximation)
            dist2 = np.sqrt(probe_coords[r]**2 + zd2)
            prop_dist[r, :] = dist1 + dist2

        prop_dist_ind = np.round(prop_dist / speed_sound * fs).astype('int')
        
        # Handle out of bounds
        max_idx = len(time_vec) - 1
        prop_dist_ind[prop_dist_ind > max_idx] = max_idx
        prop_dist_ind[prop_dist_ind < 0] = 0

        image_lines = np.zeros((n_transmit_beams, len(zd)))
        
        for q in range(n_transmit_beams):
            data_received = data_preproc[q, ...]
            scan_line = np.zeros(len(zd))
            for r in range(n_probe_channels):
                v = data_received[r, :]
                # Vectorized indexing
                scan_line = scan_line + v[prop_dist_ind[r, :]]
            image_lines[q, :] = scan_line
            
    else:
        raise ValueError(f"Unknown beamforming method: {method}")
        
    return image_lines

def run_inversion(raw_data_path, params, method='dynamic'):
    """
    Orchestrates the full reconstruction pipeline:
    1. Load & Preprocess
    2. Forward Operator (Beamforming)
    3. Post-processing (Envelope, Log Compress, Scan Convert)
    
    Returns:
        final_image: The processed B-mode image (uint8)
        x_grid: X coordinates of image
        z_grid: Z coordinates of image
    """
    
    # 1. Load and Preprocess
    data_apod, t2, xd_probe, xd_beams = load_and_preprocess_data(raw_data_path, params)
    
    # 2. Forward Operator (Beamforming)
    logger.info(f"Running beamforming method: {method}")
    rf_image_lines = forward_operator(data_apod, t2, xd_probe, params, method=method)
    
    # 3. Post-processing
    speed_sound = params['SPEED_SOUND']
    n_transmit_beams = params['N_TRANSMIT_BEAMS']
    
    # Calculate depth vector z corresponding to t2
    z = t2 * speed_sound / 2
    
    # Truncate near field ( < 5mm )
    f = np.where(z < 5e-3)[0]
    if len(f) > 0:
        cut_idx = f[-1] + 1
        z_trunc = np.delete(z, f)
        im_trunc = rf_image_lines[:, cut_idx:]
    else:
        z_trunc = z
        im_trunc = rf_image_lines

    # Envelope Detection
    im_env = np.zeros_like(im_trunc)
    for m in range(n_transmit_beams):
        im_env[m, :] = envel_detect(im_trunc[m, :])

    # Log Compression
    DR = 35 # Dynamic Range
    image_log = log_compress(im_env, DR, 0, 0)

    # Scan Conversion
    image_sc, z_sc, x_sc = scan_convert(image_log, xd_beams, z_trunc)
    
    # Normalize to 0-255 for display/metrics
    image_sc_norm = np.round(255 * image_sc / DR)
    image_sc_norm[image_sc_norm < 0] = 0
    image_sc_norm[image_sc_norm > 255] = 255
    image_final = image_sc_norm.astype('uint8').T  # Transpose to match (Depth, Width) for display
    
    return image_final, x_sc, z_sc
