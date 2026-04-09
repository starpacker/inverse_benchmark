import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(spikes: np.ndarray, gamma: float, baseline: float) -> np.ndarray:
    """
    AR(1) forward model: c[t] = gamma * c[t-1] + s[t], F[t] = c[t] + baseline
    
    Parameters
    ----------
    spikes : array of shape (n_frames,) or (n_neurons, n_frames)
        Spike train(s)
    gamma : float
        AR(1) decay parameter = exp(-dt / tau_decay)
    baseline : float
        Fluorescence baseline
    
    Returns
    -------
    fluorescence : array of same shape as spikes
        Predicted fluorescence trace(s)
    """
    if spikes.ndim == 1:
        n_frames = len(spikes)
        c = np.zeros(n_frames)
        c[0] = spikes[0]
        for t in range(1, n_frames):
            c[t] = gamma * c[t - 1] + spikes[t]
        return c + baseline
    else:
        # Handle multiple neurons
        n_neurons, n_frames = spikes.shape
        result = np.zeros_like(spikes)
        for i in range(n_neurons):
            c = np.zeros(n_frames)
            c[0] = spikes[i, 0]
            for t in range(1, n_frames):
                c[t] = gamma * c[t - 1] + spikes[i, t]
            result[i] = c + baseline
        return result
