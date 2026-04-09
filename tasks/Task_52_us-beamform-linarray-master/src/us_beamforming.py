
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp2d
import h5py
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
N_TRANSMIT_BEAMS = 96
N_PROBE_CHANNELS = 32
TRANSMIT_FREQ = 1.5e6
TRANSMIT_FOCAL_DEPTH = 20e-3
SPEED_SOUND = 1540
ARRAY_PITCH = 2 * 1.8519e-4
SAMPLE_RATE = 27.72e6
TIME_OFFSET = 1.33e-6

def arange2(start, stop=None, step=1):
    """Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop is None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a

def get_tgc(alpha0, prop_dist):
    """Time-gain compensation"""
    n = 1  # approx. 1 for soft tissue
    alpha = alpha0 * (TRANSMIT_FREQ * 1e-6)**n
    tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    return tgc_gain

def preproc(data, t, xd):
    """Preprocessing: TGC, filtering, interpolation, apodization"""
    fs = 1 / (t[1] - t[0])
    record_length = data.shape[2]
    a0 = 0.4

    # TGC
    zd = t * SPEED_SOUND / 2
    zd2 = zd**2
    dist1 = zd
    tgc = np.zeros((N_PROBE_CHANNELS, record_length))
    for r in range(N_PROBE_CHANNELS):
        dist2 = np.sqrt(xd[r]**2 + zd2)
        prop_dist = dist1 + dist2
        tgc[r, :] = get_tgc(a0, prop_dist)

    data_amp = np.zeros(data.shape)
    for m in range(N_TRANSMIT_BEAMS):
        data_amp[m, :, :] = data[m, :, :] * tgc

    # Filtering
    filt_ord = 201
    lc, hc = 0.5e6, 2.5e6
    lc = lc / (fs / 2)
    hc = hc / (fs / 2)
    B = signal.firwin(filt_ord, [lc, hc], pass_zero=False)

    # Interpolation
    interp_fact = 4
    fs_new = fs * interp_fact
    record_length2 = record_length * interp_fact
    
    # Apodization
    try:
        apod_win = signal.tukey(N_PROBE_CHANNELS)
    except AttributeError:
        # Fallback for some scipy versions or import structures
        try:
            from scipy.signal.windows import tukey
            apod_win = tukey(N_PROBE_CHANNELS)
        except ImportError:
            # Fallback to simple window if tukey is missing
            apod_win = np.hanning(N_PROBE_CHANNELS) # Close approximation or use ones
            # apod_win = np.ones(N_PROBE_CHANNELS)


    data_apod = np.zeros((N_TRANSMIT_BEAMS, N_PROBE_CHANNELS, record_length2))
    
    # Process
    for m in range(N_TRANSMIT_BEAMS):
        for n in range(N_PROBE_CHANNELS):
            w = data_amp[m, n, :]
            data_filt = signal.lfilter(B, 1, w)
            data_interp = signal.resample_poly(data_filt, interp_fact, 1)
            data_apod[m, n, :] = apod_win[n] * data_interp

    # New time vector
    freqs, delay = signal.group_delay((B, 1))
    delay = int(delay[0]) * interp_fact
    t2 = np.arange(record_length2) / fs_new + t[0] - delay / fs_new

    # Remove signal before t=0
    f = np.where(t2 < 0)[0]
    t2 = np.delete(t2, f)
    data_apod = data_apod[:, :, f[-1]+1:]

    return data_apod, t2

def beamform(data, t, xd, receive_focus):
    """Delay-and-sum beamforming with fixed focus"""
    Rf = receive_focus
    fs = 1 / (t[1] - t[0])
    delay_ind = np.zeros(N_PROBE_CHANNELS, dtype=int)
    for r in range(N_PROBE_CHANNELS):
        delay = Rf / SPEED_SOUND * (np.sqrt((xd[r] / Rf)**2 + 1) - 1)
        delay_ind[r] = int(round(delay * fs))
    max_delay = np.max(delay_ind)

    waveform_length = data.shape[2]
    image = np.zeros((N_TRANSMIT_BEAMS, waveform_length))
    for q in range(N_TRANSMIT_BEAMS):
        scan_line = np.zeros(waveform_length + max_delay)
        for r in range(N_PROBE_CHANNELS):
            delay_pad = np.zeros(delay_ind[r])
            fill_pad = np.zeros(len(scan_line) - waveform_length - delay_ind[r])
            waveform = data[q, r, :]
            # Fix concatenation shapes if needed, but original logic seems ok
            scan_line = scan_line + np.concatenate((fill_pad, waveform, delay_pad))
        image[q, :] = scan_line[max_delay:]
    return image

def beamform_df(data, t, xd):
    """Dynamic focusing beamforming"""
    fs = 1 / (t[2] - t[1])
    zd = t * SPEED_SOUND / 2
    zd2 = zd**2
    prop_dist = np.zeros((N_PROBE_CHANNELS, len(zd)))
    for r in range(N_PROBE_CHANNELS):
        dist1 = zd
        dist2 = np.sqrt(xd[r]**2 + zd2)
        prop_dist[r, :] = dist1 + dist2

    prop_dist_ind = np.round(prop_dist / SPEED_SOUND * fs).astype('int')
    
    # Handle out of bounds
    oob_inds = np.where(prop_dist_ind >= len(t))
    prop_dist_ind[oob_inds[0], oob_inds[1]] = len(t) - 1

    image = np.zeros((N_TRANSMIT_BEAMS, len(zd)))
    for q in range(N_TRANSMIT_BEAMS):
        data_received = data[q, ...]
        scan_line = np.zeros(len(zd))
        for r in range(N_PROBE_CHANNELS):
            v = data_received[r, :]
            scan_line = scan_line + v[prop_dist_ind[r, :]]
        image[q, :] = scan_line
    return image

def envel_detect(scan_line, t, method='hilbert'):
    """Envelope detection"""
    if method == 'hilbert':
        envelope = np.abs(signal.hilbert(scan_line))
    return envelope

def log_compress(data, dynamic_range, reject_level, bright_gain):
    """Log compression"""
    xd_b = 20 * np.log10(data + 1)
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
    data = data[:, 0:-1:decim_fact]
    zb = zb[0:-1:decim_fact]

    interp_func = interp2d(zb, xb, data, kind='linear')
    dz = zb[1] - zb[0]
    xnew = arange2(xb[0], xb[-1] + dz, dz)
    znew = zb
    image_sC = interp_func(znew, xnew)
    return image_sC, znew, xnew

def calculate_metrics(img_pred, img_ref):
    """Calculate PSNR and RMSE between two images"""
    # Ensure they are same size
    if img_pred.shape != img_ref.shape:
        # Resize pred to ref
        # Using simple interpolation or just cropping if close
        # For this script, let's assume they are scan converted to same grid
        pass
    
    mse = np.mean((img_pred - img_ref) ** 2)
    if mse == 0:
        return float('inf'), 0
    
    data_range = np.max([np.max(img_pred), np.max(img_ref)]) - np.min([np.min(img_pred), np.min(img_ref)])
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    rmse = np.sqrt(mse)
    
    return psnr, rmse

def main():
    data_path = 'example_us_bmode_sensor_data.h5'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info("Loading data...")
    with h5py.File(data_path, 'r') as h5f:
        sensor_data = h5f['dataset_1'][:]

    logger.info(f"Data shape: {sensor_data.shape}")

    record_length = sensor_data.shape[2]
    time_vec = np.arange(record_length) / SAMPLE_RATE - TIME_OFFSET
    xd = np.arange(N_PROBE_CHANNELS) * ARRAY_PITCH
    xd = xd - np.max(xd) / 2

    logger.info("Preprocessing...")
    preproc_data, time_shifted = preproc(sensor_data, time_vec, xd)
    logger.info(f"Preprocessed shape: {preproc_data.shape}")

    # Reconstruction 1: Fixed Focus
    logger.info("Reconstruction 1: Fixed Focus Beamforming...")
    receive_focus = 30e-3
    image_bf = beamform(preproc_data, time_shifted, xd, receive_focus)

    # Reconstruction 2: Dynamic Focusing (High Quality)
    logger.info("Reconstruction 2: Dynamic Focusing Beamforming...")
    start_time = time.time()
    image_df = beamform_df(preproc_data, time_shifted, xd)
    logger.info(f"Dynamic focusing took {time.time() - start_time:.2f}s")

    # Post-processing
    logger.info("Post-processing (Envelope, Log Compress, Scan Convert)...")
    z = time_shifted * SPEED_SOUND / 2
    xd2 = np.arange(N_TRANSMIT_BEAMS) * ARRAY_PITCH
    xd2 = xd2 - np.max(xd2) / 2

    images_to_proc = {'Fixed Focus': image_bf, 'Dynamic Focus': image_df}
    processed_images = {}
    
    # Common axes for scan conversion
    x_sc_common = None
    z_sc_common = None

    for name, im in images_to_proc.items():
        # Truncate near field
        f = np.where(z < 5e-3)[0]
        z_trunc = np.delete(z, f)
        im_trunc = im[:, f[-1]+1:]

        # Envelope
        for m in range(N_TRANSMIT_BEAMS):
            im_trunc[m, :] = envel_detect(im_trunc[m, :], 2*z_trunc/SPEED_SOUND)

        # Log Compress
        DR = 35
        image_log = log_compress(im_trunc, DR, 0, 0)

        # Scan Convert
        image_sc, z_sc, x_sc = scan_convert(image_log, xd2, z_trunc)
        
        # Normalize to 0-255
        image_sc_norm = np.round(255 * image_sc / DR)
        image_sc_norm[image_sc_norm < 0] = 0
        image_sc_norm[image_sc_norm > 255] = 255
        image_final = image_sc_norm.astype('uint8').T # Transpose for display
        
        processed_images[name] = image_final
        if x_sc_common is None:
            x_sc_common = x_sc
            z_sc_common = z_sc

    # Evaluation
    # Using Dynamic Focus as reference
    ref = processed_images['Dynamic Focus']
    target = processed_images['Fixed Focus']
    
    psnr_val, rmse_val = calculate_metrics(target, ref)
    print(f"\nEvaluation Results (Fixed Focus vs Dynamic Focus Reference):")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"RMSE: {rmse_val:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.imshow(target, extent=[x_sc_common[0]*1e3, x_sc_common[-1]*1e3, z_sc_common[-1]*1e3, z_sc_common[0]*1e3], cmap='gray')
    ax.set_title(f"Fixed Focus (PSNR: {psnr_val:.2f} dB)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    
    ax = axes[1]
    ax.imshow(ref, extent=[x_sc_common[0]*1e3, x_sc_common[-1]*1e3, z_sc_common[-1]*1e3, z_sc_common[0]*1e3], cmap='gray')
    ax.set_title("Dynamic Focus (Reference)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png')
    logger.info("Result saved to reconstruction_comparison.png")

if __name__ == "__main__":
    main()
