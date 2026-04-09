import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm
import os


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


def get_absorption_spectra(wavelengths):
    """
    Get absorption spectra for Hb and HbO2 at given wavelengths.
    Returns interpolated values based on reference data.
    """
    ref_wavelengths = np.array([700, 730, 760, 800, 850, 900])
    hb_ref = np.array([100, 80, 60, 40, 30, 20])
    hbo2_ref = np.array([30, 40, 50, 60, 70, 80])
    
    from scipy.interpolate import interp1d
    
    hb_interp = interp1d(ref_wavelengths, hb_ref, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    hbo2_interp = interp1d(ref_wavelengths, hbo2_ref, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    
    hb = hb_interp(wavelengths)
    hbo2 = hbo2_interp(wavelengths)
    
    return hb, hbo2


def load_and_preprocess_data(filename, lp_filter=7e6, hp_filter=5e3):
    """
    Load data from HDF5 file and preprocess it.
    
    Steps:
    1. Load raw data, geometry, wavelengths, speed of sound from HDF5
    2. Apply bandpass filtering
    3. Apply Hilbert transform and extract imaginary part
    
    Returns:
        dict containing processed_data, fs, geometry, wavelengths, speed_of_sound
    """
    import patato as pat
    
    print(f"Loading data from {filename}...")
    padata = pat.PAData.from_hdf5(filename)[0:1]
    
    time_series = padata.get_time_series()
    raw_data = time_series.raw_data
    fs = time_series.attributes["fs"]
    
    geometry = padata.get_scan_geometry()
    wavelengths = padata.get_wavelengths()
    speed_of_sound = padata.get_speed_of_sound()
    
    print(f"Data Loaded: Shape={raw_data.shape}, FS={fs}, SOS={speed_of_sound}")
    
    if raw_data.ndim == 4 and raw_data.shape[0] == 1:
        raw_data = raw_data[0]
    
    print("Preprocessing: Bandpass Filtering...")
    filtered_data = butter_bandpass_filter(raw_data, hp_filter, lp_filter, fs, order=4)
    
    print("Preprocessing: Envelope Detection (Hilbert)...")
    analytic_signal = hilbert(filtered_data, axis=-1)
    preprocessed_signal = np.imag(analytic_signal)
    
    return {
        'processed_data': preprocessed_signal,
        'fs': fs,
        'geometry': np.array(geometry),
        'wavelengths': np.array(wavelengths),
        'speed_of_sound': speed_of_sound
    }


def forward_operator(x, geometry, fs, speed_of_sound, n_time_samples):
    """
    Forward operator: Simulates photoacoustic signal acquisition.
    
    Given an initial pressure distribution x, computes the expected
    time-domain signals at each detector position.
    
    This implements the forward model for photoacoustic tomography:
    For each detector, the signal at time t corresponds to pressure
    contributions from points at distance r = c * t from the detector.
    
    Args:
        x: Initial pressure distribution, shape (nz, ny, nx)
        geometry: Detector positions, shape (n_det, 3)
        fs: Sampling frequency
        speed_of_sound: Speed of sound in medium
        n_time_samples: Number of time samples to generate
        
    Returns:
        y_pred: Predicted signals, shape (n_det, n_time_samples)
    """
    nz, ny, nx = x.shape
    n_det = geometry.shape[0]
    
    lx = 0.025
    ly = 0.025
    
    xs = np.linspace(-lx/2, lx/2, nx)
    ys = np.linspace(-ly/2, ly/2, ny)
    zs = np.array([0.0])
    
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    
    dl = speed_of_sound / fs
    
    y_pred = np.zeros((n_det, n_time_samples))
    
    for i_det in range(n_det):
        det_pos = geometry[i_det]
        
        dist = np.sqrt((X - det_pos[0])**2 + (Y - det_pos[1])**2 + (Z - det_pos[2])**2)
        
        sample_idx = (dist / dl).astype(int)
        
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = sample_idx[iz, iy, ix]
                    if 0 <= idx < n_time_samples:
                        y_pred[i_det, idx] += x[iz, iy, ix]
    
    return y_pred


def run_inversion(processed_data, geometry, fs, speed_of_sound, n_pixels, field_of_view):
    """
    Run the inversion (reconstruction) using Delay-and-Sum Backprojection.
    
    Also performs spectral unmixing and sO2 calculation.
    
    Args:
        processed_data: Preprocessed signal data, shape (n_wl, n_det, n_time)
        geometry: Detector positions, shape (n_det, 3)
        fs: Sampling frequency
        speed_of_sound: Speed of sound
        n_pixels: Tuple (nx, ny, nz) for reconstruction grid
        field_of_view: Tuple (lx, ly, lz) for physical dimensions
        
    Returns:
        dict containing reconstruction, concentrations, and so2
    """
    signal = processed_data
    
    nx, ny, nz = n_pixels
    lx, ly, lz = field_of_view
    
    xs = np.linspace(-lx/2, lx/2, nx)
    ys = np.linspace(-ly/2, ly/2, ny)
    zs = np.array([0.0])
    
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    
    n_wl = signal.shape[0]
    n_det = signal.shape[1]
    
    reconstruction = np.zeros((n_wl, nz, ny, nx))
    
    dl = speed_of_sound / fs
    
    print(f"Reconstructing {n_wl} wavelengths...")
    
    for i_wl in range(n_wl):
        print(f"  Wavelength {i_wl+1}/{n_wl}...")
        sig_wl = signal[i_wl]
        
        for i_det in tqdm(range(n_det), leave=False):
            det_pos = geometry[i_det]
            
            dist = np.sqrt((X - det_pos[0])**2 + (Y - det_pos[1])**2 + (Z - det_pos[2])**2)
            
            sample_idx = (dist / dl).astype(int)
            
            valid_mask = (sample_idx >= 0) & (sample_idx < sig_wl.shape[-1])
            
            reconstruction[i_wl][valid_mask] += sig_wl[i_det, sample_idx[valid_mask]]
    
    return {
        'reconstruction': reconstruction
    }


def perform_spectral_unmixing(reconstruction, wavelengths):
    """
    Perform linear spectral unmixing to estimate Hb and HbO2 concentrations.
    
    Args:
        reconstruction: Reconstructed images, shape (n_wl, nz, ny, nx)
        wavelengths: Array of wavelengths used
        
    Returns:
        concentrations: Array of shape (2, nz, ny, nx) where [0] is Hb, [1] is HbO2
    """
    hb, hbo2 = get_absorption_spectra(wavelengths)
    
    E = np.vstack([hb, hbo2]).T
    
    print("Spectral Unmixing...")
    n_wl, nz, ny, nx = reconstruction.shape
    S = reconstruction.reshape(n_wl, -1)
    
    E_inv = np.linalg.pinv(E)
    C = E_inv @ S
    
    concentrations = C.reshape(2, nz, ny, nx)
    
    return concentrations


def calculate_so2(concentrations):
    """
    Calculate oxygen saturation sO2 = HbO2 / (Hb + HbO2).
    
    Args:
        concentrations: Array of shape (2, nz, ny, nx)
        
    Returns:
        so2: Oxygen saturation map, shape (nz, ny, nx)
    """
    hb = concentrations[0]
    hbo2 = concentrations[1]
    
    total_hb = hb + hbo2
    mask = total_hb > (0.1 * np.max(total_hb))
    
    so2 = np.zeros_like(hbo2)
    so2[mask] = hbo2[mask] / total_hb[mask]
    
    so2 = np.clip(so2, 0, 1)
    
    return so2


def evaluate_results(reconstruction, so2, output_file="pat_result.png"):
    """
    Evaluate and visualize the reconstruction and sO2 results.
    
    Args:
        reconstruction: Reconstructed images, shape (n_wl, nz, ny, nx)
        so2: Oxygen saturation map, shape (nz, ny, nx)
        output_file: Path to save the output figure
        
    Returns:
        mean_so2: Mean sO2 value in the ROI
    """
    recon_img = np.mean(reconstruction, axis=0)[0]
    so2_img = so2[0]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(recon_img.T, cmap='gray', origin='lower')
    plt.title("Reconstruction (Mean WL)")
    plt.colorbar(label="PA Signal")
    
    plt.subplot(1, 2, 2)
    plt.imshow(so2_img.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.title("sO2 Estimation")
    plt.colorbar(label="sO2")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
    
    valid_so2 = so2_img[so2_img > 0]
    if len(valid_so2) > 0:
        mean_so2 = np.mean(valid_so2)
    else:
        mean_so2 = 0.0
    
    print(f"Mean sO2 in ROI: {mean_so2:.4f}")
    
    recon_max = np.max(reconstruction)
    recon_min = np.min(reconstruction)
    print(f"Reconstruction range: [{recon_min:.4f}, {recon_max:.4f}]")
    
    return mean_so2


if __name__ == "__main__":
    filename = "./dataset/invivo_oe.hdf5"
    
    n_pixels = (100, 100, 1)
    field_of_view = (0.025, 0.025, 0)
    lp_filter = 7e6
    hp_filter = 5e3
    output_file = "pat_result.png"
    
    print("Step 1: Loading and Preprocessing Data...")
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
    else:
        data_dict = load_and_preprocess_data(filename, lp_filter=lp_filter, hp_filter=hp_filter)
        
        processed_data = data_dict['processed_data']
        geometry = data_dict['geometry']
        fs = data_dict['fs']
        speed_of_sound = data_dict['speed_of_sound']
        wavelengths = data_dict['wavelengths']
        
        print("Step 2: Running Inversion (Reconstruction)...")
        inversion_result = run_inversion(
            processed_data=processed_data,
            geometry=geometry,
            fs=fs,
            speed_of_sound=speed_of_sound,
            n_pixels=n_pixels,
            field_of_view=field_of_view
        )
        
        reconstruction = inversion_result['reconstruction']
        
        print("Step 3: Spectral Unmixing...")
        concentrations = perform_spectral_unmixing(reconstruction, wavelengths)
        
        print("Step 4: Calculating sO2...")
        so2 = calculate_so2(concentrations)
        
        print("Step 5: Evaluating Results...")
        mean_so2 = evaluate_results(reconstruction, so2, output_file=output_file)
        
        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")