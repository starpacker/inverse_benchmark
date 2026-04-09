import matplotlib

matplotlib.use('Agg')

import numpy as np

def forward_operator(spikes, tau, fs, amplitude=1.0):
    """
    Forward model: convolve spike train with exponential calcium kernel.
    
    F(t) = sum_i amplitude * exp(-(t - t_i) / tau)
    
    Equivalently in AR(1) form: c_t = g * c_{t-1} + a * s_t
    
    Parameters
    ----------
    spikes : np.ndarray, shape (T,) or (N, T)
        Spike train(s).
    tau : float
        Calcium decay time constant in seconds.
    fs : float
        Sampling rate in Hz.
    amplitude : float
        Spike amplitude.
    
    Returns
    -------
    calcium : np.ndarray
        Predicted calcium trace(s), same shape as spikes.
    """
    g = np.exp(-1.0 / (tau * fs))
    
    if spikes.ndim == 1:
        T = len(spikes)
        calcium = np.zeros(T, dtype=np.float64)
        calcium[0] = amplitude * spikes[0]
        for t in range(1, T):
            calcium[t] = g * calcium[t - 1] + amplitude * spikes[t]
        return calcium
    else:
        N, T = spikes.shape
        calcium = np.zeros((N, T), dtype=np.float64)
        for i in range(N):
            calcium[i, 0] = amplitude * spikes[i, 0]
            for t in range(1, T):
                calcium[i, t] = g * calcium[i, t - 1] + amplitude * spikes[i, t]
        return calcium
