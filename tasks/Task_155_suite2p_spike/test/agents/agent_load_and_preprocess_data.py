import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.ndimage import gaussian_filter1d

def generate_spike_train(n_frames, spike_rate, fs, rng):
    """
    Generate a Poisson spike train.
    """
    prob_per_frame = spike_rate / fs
    spikes = (rng.random(n_frames) < prob_per_frame).astype(np.float64)
    return spikes

def load_and_preprocess_data(n_neurons, n_frames, spike_rate, fs, tau, amplitude, baseline, noise_std, seed=42):
    """
    Generate synthetic calcium imaging data and preprocess it.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate.
    n_frames : int
        Number of time frames.
    spike_rate : float
        Average spike rate in Hz.
    fs : float
        Sampling rate in Hz.
    tau : float
        Calcium decay time constant in seconds.
    amplitude : float
        Spike amplitude.
    baseline : float
        Baseline fluorescence F0.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed.
    
    Returns
    -------
    data_dict : dict
        Contains:
        - 'true_spikes': (n_neurons, n_frames) ground truth spike trains
        - 'calcium_traces': (n_neurons, n_frames) noisy fluorescence
        - 'clean_calcium': (n_neurons, n_frames) noise-free calcium
        - 'F_corrected': (n_neurons, n_frames) baseline-corrected fluorescence
        - 'params': dict of simulation parameters
    """
    rng = np.random.default_rng(seed)
    
    true_spikes = np.zeros((n_neurons, n_frames), dtype=np.float64)
    calcium_traces = np.zeros((n_neurons, n_frames), dtype=np.float64)
    clean_calcium = np.zeros((n_neurons, n_frames), dtype=np.float64)
    
    g = np.exp(-1.0 / (tau * fs))
    
    for i in range(n_neurons):
        # Generate spike train
        true_spikes[i] = generate_spike_train(n_frames, spike_rate, fs, rng)
        
        # Forward model: AR(1) convolution
        T = n_frames
        clean_calcium[i, 0] = amplitude * true_spikes[i, 0]
        for t in range(1, T):
            clean_calcium[i, t] = g * clean_calcium[i, t - 1] + amplitude * true_spikes[i, t]
        
        # Add noise
        noise = rng.normal(0, noise_std, T)
        calcium_traces[i] = baseline + clean_calcium[i] + noise
    
    # Preprocess: baseline correction
    F_corrected = np.zeros_like(calcium_traces)
    for i in range(n_neurons):
        smoothed = gaussian_filter1d(calcium_traces[i], sigma=fs * 0.1)
        win = int(fs * 30)
        baseline_est = np.array([
            np.min(smoothed[max(0, j - win):j + win + 1])
            for j in range(n_frames)
        ])
        F_corrected[i] = calcium_traces[i] - baseline_est
        F_corrected[i] = np.maximum(F_corrected[i], 0)
    
    params = {
        'n_neurons': n_neurons,
        'n_frames': n_frames,
        'spike_rate': spike_rate,
        'fs': fs,
        'tau': tau,
        'amplitude': amplitude,
        'baseline': baseline,
        'noise_std': noise_std,
        'seed': seed,
    }
    
    data_dict = {
        'true_spikes': true_spikes,
        'calcium_traces': calcium_traces,
        'clean_calcium': clean_calcium,
        'F_corrected': F_corrected,
        'params': params,
    }
    
    return data_dict
