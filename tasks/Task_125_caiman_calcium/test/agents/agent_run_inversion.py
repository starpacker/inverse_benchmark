import numpy as np

import matplotlib

matplotlib.use('Agg')

def run_inversion(
    fluorescence_noisy: np.ndarray,
    dt: float,
    tau_decay: float,
    lambda_l1: float,
    max_iter: int = 500,
    tol: float = 1e-6
) -> dict:
    """
    Run spike deconvolution using OASIS AR(1) algorithm with ISTA fallback.
    
    Parameters
    ----------
    fluorescence_noisy : array of shape (n_neurons, n_frames)
        Noisy fluorescence traces
    dt : float
        Time step (seconds per frame)
    tau_decay : float
        Calcium decay time constant
    lambda_l1 : float
        L1 regularization weight
    max_iter : int
        Maximum ISTA iterations (for fallback)
    tol : float
        Convergence tolerance
    
    Returns
    -------
    dict containing:
        - estimated_spikes: array of shape (n_neurons, n_frames)
        - reconstructed_fluorescence: array of shape (n_neurons, n_frames)
    """
    
    def oasis_ar1(y_data, gamma_val, lambda_val=0.0, s_min=0.0):
        """
        OASIS (Online Active Set method to Infer Spikes) for AR(1) calcium model.
        """
        y = y_data.copy()
        n = len(y)
        
        # Pool data structure: list of [value, weight, start_time, length]
        pools = []
        
        # Initialize: each time point is its own pool
        for i in range(n):
            pools.append([y[i] - lambda_val, 1.0, i, 1])
        
        # Forward pass: merge pools to satisfy constraints
        idx = 0
        while idx < len(pools):
            while idx > 0:
                prev = pools[idx - 1]
                curr = pools[idx]
                
                # Value at end of previous pool
                prev_end_val = prev[0] * (gamma_val ** prev[3])
                
                # Check constraint: c[t] >= gamma * c[t-1] + s_min
                if prev_end_val > curr[0] + s_min:
                    # Merge: combine pools
                    w_prev = prev[1]
                    w_curr = curr[1]
                    
                    # Compute merged value using weighted average
                    g_pow = gamma_val ** prev[3]
                    new_w = w_prev + w_curr * g_pow * g_pow
                    new_v = (prev[0] * w_prev + curr[0] * w_curr * g_pow) / new_w
                    
                    pools[idx - 1] = [new_v, new_w, prev[2], prev[3] + curr[3]]
                    pools.pop(idx)
                    idx -= 1
                else:
                    break
            idx += 1
        
        # Reconstruct c and s from pools
        c = np.zeros(n)
        s = np.zeros(n)
        
        for pool in pools:
            v, w, t, l = pool
            for j in range(int(l)):
                c[t + j] = max(v * (gamma_val ** j), 0)
            if t > 0:
                s[t] = max(c[t] - gamma_val * c[t - 1], 0)
            else:
                s[t] = max(c[t], 0)
        
        return c, s
    
    def deconvolve_single_neuron(fluorescence, gamma_val, lambda_val):
        """Deconvolve a single neuron's fluorescence trace."""
        n = len(fluorescence)
        
        # Estimate baseline robustly (lower percentile of fluorescence)
        est_baseline = np.percentile(fluorescence, 15)
        
        # Subtract baseline
        y = fluorescence - est_baseline
        
        # Normalize y to [0, ~1] range for numerical stability
        y_scale = np.percentile(np.abs(y), 99) + 1e-8
        y_norm = y / y_scale
        
        # Run OASIS
        c_oasis, s_oasis = oasis_ar1(y_norm, gamma_val, lambda_val=lambda_val * 0.5)
        
        # If OASIS doesn't work well, use simple non-negative deconvolution
        if np.sum(s_oasis > 0) < 5:
            s_wiener = np.zeros(n)
            s_wiener[0] = max(y_norm[0], 0)
            for t in range(1, n):
                s_wiener[t] = max(y_norm[t] - gamma_val * y_norm[t - 1], 0)
            # Threshold small values
            threshold = np.std(s_wiener) * 0.5
            s_wiener[s_wiener < threshold] = 0
            return s_wiener, est_baseline, y_scale
        else:
            return s_oasis, est_baseline, y_scale
    
    n_neurons, n_frames = fluorescence_noisy.shape
    gamma = np.exp(-dt / tau_decay)
    
    all_est_spikes = []
    all_reconstructed_fluor = []
    
    for i in range(n_neurons):
        est_spikes, est_baseline, y_scale = deconvolve_single_neuron(
            fluorescence_noisy[i], gamma, lambda_l1
        )
        all_est_spikes.append(est_spikes)
        
        # Reconstruct fluorescence from estimated spikes using AR(1) model
        recon_calcium = np.zeros(n_frames)
        recon_calcium[0] = est_spikes[0]
        for t in range(1, n_frames):
            recon_calcium[t] = gamma * recon_calcium[t - 1] + est_spikes[t]
        recon_fluor = recon_calcium * y_scale + est_baseline
        all_reconstructed_fluor.append(recon_fluor)
    
    return {
        'estimated_spikes': np.array(all_est_spikes),
        'reconstructed_fluorescence': np.array(all_reconstructed_fluor),
    }
