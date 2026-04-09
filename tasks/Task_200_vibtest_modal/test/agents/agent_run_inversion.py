import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.signal import find_peaks as sp_find_peaks

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(omega, H_noisy, n_modes=3):
    """
    Extract modal parameters from noisy FRF using peak-picking + half-power bandwidth.
    
    Args:
        omega: frequency array (rad/s)
        H_noisy: noisy FRF measurements (complex)
        n_modes: number of modes to extract
    
    Returns:
        dict containing:
            - freq_est: estimated natural frequencies (rad/s)
            - zeta_est: estimated damping ratios
            - phi_est: estimated mode shapes
    """
    print("\n[5] Extracting modal parameters...")
    
    n_dof = H_noisy.shape[1]
    
    # Peak picking: use SUM of all FRF magnitudes (Mode Indicator Function)
    mif = np.zeros(len(omega))
    for j in range(n_dof):
        mif += np.abs(H_noisy[:, j])**2
    mif_db = 10 * np.log10(mif + 1e-30)
    
    # Find peaks with adaptive prominence
    dyn_range = np.max(mif_db) - np.min(mif_db)
    prominence = 0.15 * dyn_range
    min_dist = max(int(0.03 * len(omega)), 10)
    
    peak_indices, props = sp_find_peaks(mif_db, prominence=prominence, distance=min_dist)
    
    # Sort by frequency and take first n_modes
    peak_indices = np.sort(peak_indices)[:n_modes]
    
    if len(peak_indices) < n_modes:
        # Lower threshold
        prominence = 0.05 * dyn_range
        peak_indices, _ = sp_find_peaks(mif_db, prominence=prominence, distance=min_dist)
        peak_indices = np.sort(peak_indices)[:n_modes]
    
    print(f"  Peak frequencies (Hz): {omega[peak_indices] / (2*np.pi)}")
    
    freq_est = np.zeros(n_modes)
    zeta_est = np.zeros(n_modes)
    phi_est = np.zeros((n_dof, n_modes))
    
    # Use drive-point FRF magnitude for half-power method
    mag = np.abs(H_noisy[:, 0])
    
    for r in range(n_modes):
        pk = peak_indices[r]
        omega_pk = omega[pk]
        
        # Refine peak location using parabolic interpolation
        if 1 <= pk < len(omega) - 1:
            alpha_val = mif_db[pk - 1]
            beta_val = mif_db[pk]
            gamma_val = mif_db[pk + 1]
            p = 0.5 * (alpha_val - gamma_val) / (alpha_val - 2*beta_val + gamma_val + 1e-30)
            domega = omega[1] - omega[0]
            omega_n_est = omega_pk + p * domega
        else:
            omega_n_est = omega_pk
        
        freq_est[r] = omega_n_est
        
        # Damping: half-power bandwidth method
        bw_factor = 0.15
        if r > 0:
            f_lower = max(omega_n_est * (1 - bw_factor),
                         (omega[peak_indices[r-1]] + omega_n_est) / 2)
        else:
            f_lower = max(omega_n_est * (1 - bw_factor), omega[0])
        
        if r < n_modes - 1:
            f_upper = min(omega_n_est * (1 + bw_factor),
                         (omega_n_est + omega[peak_indices[r+1]]) / 2)
        else:
            f_upper = min(omega_n_est * (1 + bw_factor), omega[-1])
        
        mask = (omega >= f_lower) & (omega <= f_upper)
        omega_band = omega[mask]
        mif_band = np.sqrt(mif[mask])  # RMS of all FRFs
        
        peak_val_mif = np.sqrt(mif[pk])
        hp_level = peak_val_mif / np.sqrt(2)
        
        # Find crossing points
        above = mif_band > hp_level
        transitions = np.where(np.diff(above.astype(int)))[0]
        
        if len(transitions) >= 2:
            # Left crossing
            idx_lo = transitions[0]
            if idx_lo + 1 < len(omega_band):
                frac = (hp_level - mif_band[idx_lo]) / (mif_band[idx_lo+1] - mif_band[idx_lo] + 1e-30)
                omega_1 = omega_band[idx_lo] + frac * (omega_band[idx_lo+1] - omega_band[idx_lo])
            else:
                omega_1 = omega_band[idx_lo]
            
            # Right crossing
            idx_hi = transitions[-1]
            if idx_hi + 1 < len(omega_band):
                frac = (hp_level - mif_band[idx_hi]) / (mif_band[idx_hi+1] - mif_band[idx_hi] + 1e-30)
                omega_2 = omega_band[idx_hi] + frac * (omega_band[idx_hi+1] - omega_band[idx_hi])
            else:
                omega_2 = omega_band[idx_hi]
            
            zeta_est[r] = abs(omega_2 - omega_1) / (2 * omega_n_est)
        else:
            zeta_est[r] = 0.01
        
        zeta_est[r] = max(zeta_est[r], 1e-4)
        
        # Mode shape: use imaginary part of FRF at resonance
        idx_res = np.argmin(np.abs(omega - freq_est[r]))
        
        bw_pts = max(int(0.005 * len(omega)), 3)
        lo = max(0, idx_res - bw_pts)
        hi = min(len(omega), idx_res + bw_pts + 1)
        
        for j in range(n_dof):
            imag_vals = np.imag(H_noisy[lo:hi, j])
            phi_est[j, r] = np.mean(imag_vals)
    
    # Normalize mode shapes
    for r in range(n_modes):
        norm = np.linalg.norm(phi_est[:, r])
        if norm > 1e-10:
            phi_est[:, r] /= norm
    
    print(f"  Identified frequencies (Hz): {freq_est / (2*np.pi)}")
    print(f"  Identified damping ratios:   {zeta_est}")
    
    return {
        'freq_est': freq_est,
        'zeta_est': zeta_est,
        'phi_est': phi_est,
    }
