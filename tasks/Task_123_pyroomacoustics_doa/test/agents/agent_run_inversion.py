import numpy as np

import matplotlib

matplotlib.use('Agg')

import pyroomacoustics as pra

np.random.seed(42)

def run_inversion(
    X: np.ndarray,
    mic_locs: np.ndarray,
    fs: int,
    nfft: int,
    c: float,
    n_sources: int,
    freq_range: list = [500, 4000]
) -> dict:
    """
    Run DOA estimation using MUSIC algorithm (and SRP-PHAT for comparison).
    
    This is the inverse problem: given multi-channel frequency-domain data,
    estimate the angular positions of acoustic sources.
    
    Args:
        X: STFT data, shape (n_channels, n_freq_bins, n_frames)
        mic_locs: Microphone positions, shape (2, n_mics)
        fs: Sampling frequency
        nfft: FFT size
        c: Speed of sound
        n_sources: Number of sources to find
        freq_range: Frequency range for DOA estimation
    
    Returns:
        Dictionary containing estimated azimuths and spatial spectra
    """
    print("\nStep 5: DOA estimation using MUSIC algorithm...")
    
    # Create MUSIC DOA estimator
    doa_music = pra.doa.MUSIC(mic_locs, fs, nfft, c=c, num_src=n_sources, dim=2)
    
    # Estimate DOA
    doa_music.locate_sources(X, freq_range=freq_range)
    
    # Extract spatial spectrum
    spatial_spectrum = doa_music.grid.values.copy()
    azimuth_grid = doa_music.grid.azimuth.copy()  # in radians [0, 2*pi)
    azimuth_grid_deg = np.degrees(azimuth_grid)
    
    # Find peaks (estimated source directions)
    peak_indices = doa_music.grid.find_peaks(k=n_sources)
    estimated_azimuths_rad = azimuth_grid[peak_indices]
    estimated_azimuths_deg = np.degrees(estimated_azimuths_rad)
    
    print(f"  Estimated azimuths (deg): {[f'{a:.1f}' for a in estimated_azimuths_deg]}")
    
    # Also run SRP-PHAT for comparison
    print("\nStep 5b: DOA estimation using SRP-PHAT (comparison)...")
    doa_srp = pra.doa.SRP(mic_locs, fs, nfft, c=c, num_src=n_sources, dim=2)
    doa_srp.locate_sources(X, freq_range=freq_range)
    srp_spectrum = doa_srp.grid.values.copy()
    srp_peaks = doa_srp.grid.find_peaks(k=n_sources)
    srp_azimuths_deg = np.degrees(azimuth_grid[srp_peaks])
    print(f"  SRP-PHAT estimated azimuths (deg): {[f'{a:.1f}' for a in srp_azimuths_deg]}")
    
    return {
        "spatial_spectrum": spatial_spectrum,
        "azimuth_grid": azimuth_grid,
        "azimuth_grid_deg": azimuth_grid_deg,
        "peak_indices": peak_indices,
        "estimated_azimuths_rad": estimated_azimuths_rad,
        "estimated_azimuths_deg": estimated_azimuths_deg,
        "srp_spectrum": srp_spectrum,
        "srp_azimuths_deg": srp_azimuths_deg,
    }
