import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.signal import fftconvolve

def load_and_preprocess_data(
    n_neurons: int,
    n_frames: int,
    dt: float,
    tau_rise: float,
    tau_decay: float,
    spike_rate: float,
    snr: float,
    baseline: float,
    kernel_duration: float = 5.0,
    seed: int = 42
) -> dict:
    """
    Load and preprocess data: create calcium kernel and simulate neurons.
    
    Returns a dictionary containing:
    - kernel: calcium impulse response
    - kernel_t: time axis for kernel
    - true_spikes: ground truth spike trains (n_neurons, n_frames)
    - fluorescence_clean: clean fluorescence traces
    - fluorescence_noisy: noisy fluorescence traces
    - params: dictionary of simulation parameters
    """
    np.random.seed(seed)
    
    # Create double-exponential calcium impulse response kernel
    n_pts = int(kernel_duration / dt)
    t_kernel = np.arange(n_pts) * dt
    kernel = np.exp(-t_kernel / tau_decay) - np.exp(-t_kernel / tau_rise)
    kernel[kernel < 0] = 0
    kernel /= kernel.max()  # normalize peak to 1
    
    # Simulate neurons
    all_true_spikes = []
    all_fluorescence_clean = []
    all_fluorescence_noisy = []
    
    amp = 200.0  # peak amplitude above baseline
    
    for i in range(n_neurons):
        # Generate Poisson spike train
        spikes = np.random.poisson(spike_rate, size=n_frames).astype(np.float64)
        
        # Forward model: convolve spikes with calcium kernel
        calcium = fftconvolve(spikes, kernel, mode='full')[:n_frames]
        calcium_scaled = calcium * amp
        
        # Clean fluorescence
        fluorescence_clean = calcium_scaled + baseline
        
        # Add Gaussian noise
        signal_power = np.var(calcium_scaled)
        noise_std = np.sqrt(signal_power / snr)
        noise = np.random.randn(n_frames) * noise_std
        fluorescence_noisy = fluorescence_clean + noise
        
        all_true_spikes.append(spikes)
        all_fluorescence_clean.append(fluorescence_clean)
        all_fluorescence_noisy.append(fluorescence_noisy)
    
    all_true_spikes = np.array(all_true_spikes)
    all_fluorescence_clean = np.array(all_fluorescence_clean)
    all_fluorescence_noisy = np.array(all_fluorescence_noisy)
    
    params = {
        'n_neurons': n_neurons,
        'n_frames': n_frames,
        'dt': dt,
        'tau_rise': tau_rise,
        'tau_decay': tau_decay,
        'spike_rate': spike_rate,
        'snr': snr,
        'baseline': baseline,
    }
    
    return {
        'kernel': kernel,
        'kernel_t': t_kernel,
        'true_spikes': all_true_spikes,
        'fluorescence_clean': all_fluorescence_clean,
        'fluorescence_noisy': all_fluorescence_noisy,
        'params': params,
    }
