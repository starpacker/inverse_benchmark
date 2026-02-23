```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from tqdm import tqdm
import os

# ==============================================================================
# Helper Functions (Inlined from libraries)
# ==============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y

def envelope_detection(data):
    """Compute envelope using Hilbert transform along the last axis."""
    analytic_signal = hilbert(data, axis=-1)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def get_absorption_spectra():
    """
    Define absorption spectra for Hb and HbO2 (simplified for demo).
    In a real app, these would be loaded from a database.
    Values are approximate extinction coefficients [cm^-1 M^-1] or similar units.
    """
    # Wavelengths often used in MSOT
    wavelengths = np.array([700, 730, 760, 800, 850, 900])
    
    # Approximate values (normalized for unmixing demo)
    # Hb (Deoxyhemoglobin)
    hb = np.array([100, 80, 60, 40, 30, 20]) 
    # HbO2 (Oxyhemoglobin)
    hbo2 = np.array([30, 40, 50, 60, 70, 80])
    
    return wavelengths, hb, hbo2

# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_data(filename):
    """
    Load data from HDF5. 
    Using patato for loading to handle the specific file structure, 
    then extracting numpy arrays.
    """
    import patato as pat
    print(f"Loading data from {filename}...")
    padata = pat.PAData.from_hdf5(filename)[0:1] # Load first frame/scan
    
    # Extract raw data
    time_series = padata.get_time_series()
    raw_data = time_series.raw_data # (N_wavelengths, N_detectors, N_samples)
    fs = time_series.attributes["fs"]
    
    # Extract metadata
    geometry = padata.get_scan_geometry()
    wavelengths = padata.get_wavelengths()
    speed_of_sound = padata.get_speed_of_sound()
    
    print(f"Data Loaded: Shape={raw_data.shape}, FS={fs}, SOS={speed_of_sound}")
    
    # Squeeze if single frame
    if raw_data.ndim == 4 and raw_data.shape[0] == 1:
        raw_data = raw_data[0]
        
    return {
        'raw_data': raw_data,
        'fs': fs,
        'geometry': geometry,
        'wavelengths': wavelengths,
        'speed_of_sound': speed_of_sound,
        'padata_obj': padata # Keep reference for ROIs if needed
    }

def preprocess_data(data_dict, lp_filter=7e6, hp_filter=5e3):
    """
    Explicit preprocessing pipeline: Bandpass Filter -> Hilbert Transform (Envelope).
    """
    raw_data = data_dict['raw_data']
    fs = data_dict['fs']
    
    print("Preprocessing: Bandpass Filtering...")
    # Apply filter to each wavelength/detector
    filtered_data = butter_bandpass_filter(raw_data, hp_filter, lp_filter, fs, order=4)
    
    print("Preprocessing: Envelope Detection (Hilbert)...")
    # Note: patato's MSOTPreProcessor might do Hilbert. 
    # Usually for Backprojection, we project the signal, but envelope is often used for visualization.
    # However, standard BP often uses the raw RF data (or filtered RF). 
    # If we want to match patato's default which might use Hilbert:
    # "absolute" defaults to "imag" (if hilbert=True) or "abs".
    # Let's stick to filtered RF data for reconstruction as it preserves phase info for DAS,
    # OR if the user wants "envelope", we do it. 
    # Patato's main.py says: pre_processor = pat.MSOTPreProcessor(..., hp_filter=..., lp_filter=...)
    # Defaults in patato: hilbert=True, absolute='imag'. 
    # This corresponds to the imaginary part of the analytic signal.
    
    analytic_signal = hilbert(filtered_data, axis=-1)
    preprocessed_signal = np.imag(analytic_signal) # Using Imaginary part as per Patato default
    
    data_dict['processed_data'] = preprocessed_signal
    return data_dict

# ==============================================================================
# 2. Forward/Inverse Operator (Reconstruction)
# ==============================================================================

def backproject(data_dict, n_pixels=(333, 333, 1), field_of_view=(0.025, 0.025, 0)):
    """
    Explicit Delay-and-Sum Backprojection implementation.
    """
    signal = data_dict['processed_data'] # (N_wl, N_det, N_time)
    geometry = data_dict['geometry']
    fs = data_dict['fs']
    sos = data_dict['speed_of_sound']
    
    # Grid setup
    nx, ny, nz = n_pixels
    lx, ly, lz = field_of_view
    
    xs = np.linspace(-lx/2, lx/2, nx)
    ys = np.linspace(-ly/2, ly/2, ny)
    zs = np.array([0.0])
    
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij") # Shape (1, ny, nx)
    
    # Output shape: (N_wl, 1, ny, nx) -> We usually recon each wavelength
    n_wl = signal.shape[0]
    n_det = signal.shape[1]
    
    reconstruction = np.zeros((n_wl, nz, ny, nx))
    
    dl = sos / fs
    
    print(f"Reconstructing {n_wl} wavelengths...")
    
    for i_wl in range(n_wl):
        print(f"  Wavelength {i_wl+1}/{n_wl}...")
        sig_wl = signal[i_wl] # (N_det, N_time)
        
        # Simple loop over detectors (Vectorized over pixels)
        # To optimize, we can batch pixels or detectors. 
        # Here we loop detectors and add to image.
        
        for i_det in tqdm(range(n_det), leave=False):
            det_pos = geometry[i_det] # (x, y, z)
            
            # Distance from detector to every pixel
            dist = np.sqrt((X - det_pos[0])**2 + (Y - det_pos[1])**2 + (Z - det_pos[2])**2)
            
            # Convert distance to sample index
            sample_idx = (dist / dl).astype(int)
            
            # Valid indices
            valid_mask = (sample_idx >= 0) & (sample_idx < sig_wl.shape[-1])
            
            # Add signal to reconstruction
            # We need to handle the broadcasting carefully
            # reconstruction[i_wl, :, :, :] += sig_wl[i_det, sample_idx] where valid
            
            # Using flat indexing for speed or masked arrays
            # Let's do a safe addition
            reconstruction[i_wl][valid_mask] += sig_wl[i_det, sample_idx[valid_mask]]
            
    data_dict['reconstruction'] = reconstruction
    return reconstruction

# ==============================================================================
# 3. Spectral Unmixing & Analysis
# ==============================================================================

def spectral_unmixing(data_dict):
    """
    Linear Spectral Unmixing to find Hb and HbO2 concentrations.
    C = pinv(E) * S
    """
    recon = data_dict['reconstruction'] # (N_wl, nz, ny, nx)
    wavelengths = data_dict['wavelengths']
    
    # Get reference spectra (Interpolate to measured wavelengths)
    # Using patato's spectra for consistency
    from patato.unmixing.spectra import SPECTRA_NAMES
    
    # Hb and HbO2
    hb = SPECTRA_NAMES["Hb"].get_spectrum(wavelengths)
    hbo2 = SPECTRA_NAMES["HbO2"].get_spectrum(wavelengths)
    
    E = np.vstack([hb, hbo2]).T # Shape (N_wl, 2)
    
    print("Spectral Unmixing...")
    # Reshape recon for matrix multiplication: (N_wl, N_pixels)
    n_wl, nz, ny, nx = recon.shape
    S = recon.reshape(n_wl, -1)
    
    # Solve C = pinv(E) @ S
    # E_inv: (2, N_wl)
    E_inv = np.linalg.pinv(E)
    C = E_inv @ S # (2, N_pixels)
    
    # Reshape back
    concentrations = C.reshape(2, nz, ny, nx)
    
    data_dict['concentrations'] = concentrations # [0] is Hb, [1] is HbO2
    return concentrations

def calculate_so2(data_dict):
    """
    Calculate sO2 = HbO2 / (Hb + HbO2)
    """
    concentrations = data_dict['concentrations']
    hb = concentrations[0]
    hbo2 = concentrations[1]
    
    # Avoid division by zero
    total_hb = hb + hbo2
    mask = total_hb > (0.1 * np.max(total_hb)) # Threshold to avoid noise
    
    so2 = np.zeros_like(hbo2)
    so2[mask] = hbo2[mask] / total_hb[mask]
    
    # Clip to [0, 1]
    so2 = np.clip(so2, 0, 1)
    
    data_dict['so2'] = so2
    return so2

# ==============================================================================
# 4. Evaluation / Visualization
# ==============================================================================

def evaluate_results(data_dict, output_file="pat_result.png"):
    """
    Visualize Reconstruction and sO2.
    """
    recon = data_dict['reconstruction']
    so2 = data_dict['so2']
    
    # Take max projection or middle slice
    # Data is (N_wl, 1, ny, nx) -> squeeze to (ny, nx)
    # We visualize one wavelength (e.g., 800nm which is index ~3) or mean
    recon_img = np.mean(recon, axis=0)[0] # Mean over wavelengths, z=0
    so2_img = so2[0] # z=0
    
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
    
    # Simple metric
    mean_so2 = np.mean(so2_img[so2_img > 0])
    print(f"Mean sO2 in ROI: {mean_so2:.4f}")
    return mean_so2

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    filename = "./dataset/invivo_oe.hdf5"
    
    print("Step 1: Loading Data...")
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
    else:
        data_dict = load_data(filename)
        
        print("Step 2: Preprocessing...")
        data_dict = preprocess_data(data_dict)
        
        print("Step 3: Reconstruction (Inverse)...")
        # Use lower resolution for speed in demo if needed, or standard
        backproject(data_dict, n_pixels=(100, 100, 1), field_of_view=(0.025, 0.025, 0))
        
        print("Step 4: Unmixing & Analysis...")
        spectral_unmixing(data_dict)
        calculate_so2(data_dict)
        
        print("Step 5: Evaluation...")
        evaluate_results(data_dict)
        
        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```