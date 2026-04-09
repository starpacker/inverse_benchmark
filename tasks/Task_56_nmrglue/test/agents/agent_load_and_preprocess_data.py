import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.fft import fft, ifft, fft2, ifft2, fftshift

import nmrglue as ng

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_synthetic_peaks(n_peaks, sw_f1, sw_f2, seed=42):
    """
    Generate random peak parameters for a 2D NMR spectrum.
    """
    rng = np.random.default_rng(seed)
    peaks = []
    for i in range(n_peaks):
        peaks.append({
            "freq_f1": rng.uniform(0.15, 0.85) * sw_f1,
            "freq_f2": rng.uniform(0.15, 0.85) * sw_f2,
            "lw_f1": rng.uniform(10, 50),
            "lw_f2": rng.uniform(15, 80),
            "amplitude": rng.uniform(0.5, 2.0),
            "phase": rng.uniform(-0.1, 0.1),
        })
    return peaks

def synthesize_fid(peaks, n_f1, n_f2, sw_f1, sw_f2):
    """
    Synthesize a 2D FID from Lorentzian peaks.
    """
    dt1 = 1.0 / sw_f1
    dt2 = 1.0 / sw_f2
    t1 = np.arange(n_f1) * dt1
    t2 = np.arange(n_f2) * dt2

    fid = np.zeros((n_f1, n_f2), dtype=complex)
    for p in peaks:
        decay_f1 = np.exp(-np.pi * p["lw_f1"] * t1)
        osc_f1 = np.exp(1j * 2 * np.pi * p["freq_f1"] * t1)
        decay_f2 = np.exp(-np.pi * p["lw_f2"] * t2)
        osc_f2 = np.exp(1j * 2 * np.pi * p["freq_f2"] * t2)
        sig_f1 = p["amplitude"] * np.exp(1j * p["phase"]) * decay_f1 * osc_f1
        sig_f2 = decay_f2 * osc_f2
        fid += np.outer(sig_f1, sig_f2)

    return fid

def load_and_preprocess_data(n_f1, n_f2, sw_f1, sw_f2, n_peaks, nus_frac, noise_std, seed):
    """
    Generate synthetic 2D NMR data with NUS sampling.
    
    Parameters
    ----------
    n_f1 : int
        Number of points in indirect dimension (F1).
    n_f2 : int
        Number of points in direct dimension (F2).
    sw_f1 : float
        Spectral width F1 [Hz].
    sw_f2 : float
        Spectral width F2 [Hz].
    n_peaks : int
        Number of resonance peaks.
    nus_frac : float
        Fraction of indirect-dimension points sampled.
    noise_std : float
        Noise level relative to max FID amplitude.
    seed : int
        Random seed.
    
    Returns
    -------
    dict containing:
        fid_nus : np.ndarray
            NUS-sampled FID.
        spec_gt : np.ndarray
            Ground truth spectrum.
        fid_full : np.ndarray
            Complete FID (before NUS).
        schedule : np.ndarray
            Boolean NUS sampling mask.
    """
    print("[DATA] Generating synthetic 2D NMR peaks ...")
    peaks = generate_synthetic_peaks(n_peaks, sw_f1, sw_f2, seed)

    print(f"[DATA] Synthesising complete FID ({n_f1}×{n_f2}) ...")
    fid_full = synthesize_fid(peaks, n_f1, n_f2, sw_f1, sw_f2)

    # Ground truth spectrum: apply apodization and full 2D FFT
    fid_proc_gt = ng.proc_base.em(fid_full, lb=5.0)
    spec_gt = fftshift(fft2(fid_proc_gt)).real
    spec_gt = spec_gt / np.abs(spec_gt).max()

    # NUS schedule (random sampling of indirect dimension)
    rng = np.random.default_rng(seed + 1)
    n_sampled = max(int(n_f1 * nus_frac), 2)
    schedule = np.zeros(n_f1, dtype=bool)
    schedule[0] = True
    chosen = rng.choice(np.arange(1, n_f1), size=n_sampled - 1, replace=False)
    schedule[chosen] = True

    print(f"[DATA] NUS schedule: {schedule.sum()}/{n_f1} points "
          f"({schedule.sum()/n_f1*100:.0f}%)")

    # Add noise
    rng2 = np.random.default_rng(seed + 2)
    noise = noise_std * np.abs(fid_full).max() * (
        rng2.standard_normal(fid_full.shape) +
        1j * rng2.standard_normal(fid_full.shape)
    )
    fid_noisy = fid_full + noise

    # Apply NUS forward operator (sampling mask)
    fid_nus = forward_operator(fid_noisy, schedule)

    return {
        "fid_nus": fid_nus,
        "spec_gt": spec_gt,
        "fid_full": fid_full,
        "schedule": schedule
    }

def forward_operator(fid_full, nus_schedule):
    """
    NUS forward operator: apply sampling mask to FID.
    
    Uses nmrglue's processing pipeline for apodization of the
    direct dimension, then masks the indirect dimension according
    to the NUS schedule.
    
    Parameters
    ----------
    fid_full : np.ndarray
        Complete 2D FID (n_f1 × n_f2).
    nus_schedule : np.ndarray
        Boolean mask of sampled t1 points.
    
    Returns
    -------
    fid_nus : np.ndarray
        NUS-sampled FID (same shape, zeros at unsampled t1 rows).
    """
    # Apply nmrglue apodization to F2 (direct dimension)
    fid_proc = ng.proc_base.em(fid_full, lb=5.0)

    # Apply NUS mask to indirect dimension (F1)
    fid_nus = np.zeros_like(fid_proc)
    fid_nus[nus_schedule, :] = fid_proc[nus_schedule, :]

    return fid_nus
