import numpy as np

import matplotlib

matplotlib.use('Agg')

import pyroomacoustics as pra

np.random.seed(42)

def forward_operator(
    noisy_signals: np.ndarray,
    nfft: int
) -> np.ndarray:
    """
    Apply the forward operator: compute STFT of multi-channel recordings.
    
    This transforms time-domain microphone signals to frequency-domain
    for spatial spectrum analysis.
    
    Args:
        noisy_signals: Multi-channel recordings, shape (n_mics, n_samples)
        nfft: FFT size for STFT
    
    Returns:
        X: STFT output, shape (n_channels, n_freq_bins, n_frames)
    """
    # pra STFT expects (n_samples, n_channels) input
    X = pra.transform.stft.analysis(noisy_signals.T, nfft, nfft // 2)
    # X shape: (n_frames, n_freq_bins, n_channels)
    # DOA expects: (n_channels, n_freq_bins, n_frames)
    X = X.transpose([2, 1, 0])
    print(f"  STFT shape (channels, freq_bins, frames): {X.shape}")
    
    return X
