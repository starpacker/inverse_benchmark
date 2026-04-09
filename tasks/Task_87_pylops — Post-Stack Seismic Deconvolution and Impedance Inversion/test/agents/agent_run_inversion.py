import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def run_inversion(seismic_obs, wavelet, impedance_bg, impedance_true):
    """
    Run post-stack seismic inversion to recover reflectivity and impedance.
    
    Uses two approaches:
    1. Regularized deconvolution to estimate reflectivity, then integrate
    2. pylops PoststackLinearModelling for direct impedance inversion
    
    Args:
        seismic_obs: observed seismic data (nt, n_traces)
        wavelet: source wavelet array
        impedance_bg: background impedance model (nt, n_traces)
        impedance_true: true impedance for initial value reference
    
    Returns:
        dict containing:
            - reflectivity_inv: inverted reflectivity (nt, n_traces)
            - impedance_inv: inverted impedance from method 1
            - impedance_inv2: inverted impedance from method 2
            - impedance_final: best impedance result
    """
    from pylops.signalprocessing import Convolve1D
    from pylops import FirstDerivative
    from pylops.optimization.leastsquares import regularized_inversion
    from pylops.avo.poststack import PoststackLinearModelling
    
    nt, n_traces = seismic_obs.shape
    
    # Method 1: Regularized deconvolution for reflectivity
    print("\n[INV] Running regularized deconvolution...")
    reflectivity_inv = np.zeros_like(seismic_obs)
    
    for j in range(n_traces):
        # Forward operator: convolution with wavelet
        Cop = Convolve1D(nt, h=wavelet, offset=len(wavelet)//2)
        
        # Regularization: first derivative (promotes blocky impedance)
        D1op = FirstDerivative(nt, dtype='float64')
        
        # Regularized least-squares inversion
        # min ||Cop @ r - s_obs||² + λ ||D1 @ r||²
        r_inv = regularized_inversion(
            Cop, seismic_obs[:, j],
            [D1op],
            epsRs=[1e-1],
            **dict(iter_lim=50)
        )[0]
        
        reflectivity_inv[:, j] = r_inv
    
    print(f"  Inverted reflectivity range: [{reflectivity_inv.min():.4f}, "
          f"{reflectivity_inv.max():.4f}]")
    
    # Convert reflectivity to impedance
    print("\n[POST] Converting reflectivity to impedance...")
    initial_impedance = impedance_true[0, :].mean()
    ratio = (1 + reflectivity_inv) / (1 - reflectivity_inv + 1e-10)
    log_ratio = np.log(np.abs(ratio) + 1e-10)
    impedance_inv = initial_impedance * np.exp(np.cumsum(log_ratio, axis=0))
    
    # Method 2: pylops PoststackLinearModelling
    print("\n[INV2] Running pylops PoststackLinearModelling inversion...")
    impedance_inv2 = np.zeros_like(seismic_obs)
    
    for j in range(n_traces):
        # Create post-stack modelling operator
        PPop = PoststackLinearModelling(wavelet, nt0=nt, explicit=False)
        
        # Background log-impedance model
        m_bg = np.log(impedance_bg[:, j])
        
        # Regularization: smoothness
        D2op = FirstDerivative(nt, dtype='float64')
        
        # Inversion for log-impedance perturbation
        dm_inv = regularized_inversion(
            PPop, seismic_obs[:, j],
            [D2op],
            epsRs=[5e-1],
            **dict(iter_lim=80)
        )[0]
        
        # Reconstruct impedance
        impedance_inv2[:, j] = np.exp(m_bg + dm_inv)
    
    return {
        'reflectivity_inv': reflectivity_inv,
        'impedance_inv': impedance_inv,
        'impedance_inv2': impedance_inv2,
    }
