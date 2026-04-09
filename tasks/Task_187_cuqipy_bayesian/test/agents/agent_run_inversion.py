import warnings

import numpy as np

import matplotlib

matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from cuqi.distribution import Gaussian, GMRF, LMRF

from cuqi.problem import BayesianProblem

def run_inversion(data_dict, gmrf_precision, lmrf_precision, n_samples):
    """
    Run Bayesian inversion using CUQIpy.
    
    Performs:
    1. Define Bayesian model with GMRF prior and Gaussian likelihood.
    2. Compute MAP estimate.
    3. Draw posterior samples.
    4. Optionally try LMRF prior and compare.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data containing:
        - 'y_data': observed data
        - 'x_true': ground truth
        - 'forward_model': forward operator
        - 'dim': dimension
        - 'noise_std': noise level
    gmrf_precision : float
        Precision parameter for GMRF prior.
    lmrf_precision : float
        Precision parameter for LMRF prior.
    n_samples : int
        Number of posterior samples to draw.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'x_map_gmrf': MAP estimate with GMRF prior
        - 'x_map_lmrf': MAP estimate with LMRF prior (or None)
        - 'posterior_mean': posterior mean from GMRF
        - 'posterior_std': posterior standard deviation
        - 'lower_ci': 2.5th percentile (lower credible bound)
        - 'upper_ci': 97.5th percentile (upper credible bound)
        - 'samples': posterior samples array
        - 'n_samples': number of samples
        - 'gmrf_precision': GMRF precision used
        - 'lmrf_precision': LMRF precision used
    """
    y_data = data_dict['y_data']
    A = data_dict['forward_model']
    dim = data_dict['dim']
    noise_std = data_dict['noise_std']
    
    # ========== GMRF Prior ==========
    print(f"\nBayesian model:")
    print(f"  Prior: GMRF(0, precision={gmrf_precision})")
    print(f"  Likelihood: Gaussian(A@x, noise_var={noise_std**2})")
    
    # Define prior: GMRF
    x = GMRF(np.zeros(dim), gmrf_precision, geometry=A.domain_geometry)
    
    # Define likelihood: Gaussian
    y = Gaussian(mean=A @ x, cov=noise_std**2)
    
    # Create Bayesian problem
    BP = BayesianProblem(y, x).set_data(y=y_data)
    
    # Compute MAP estimate
    print("\nComputing MAP estimate...")
    x_map_gmrf = np.asarray(BP.MAP())
    
    # Sample posterior
    print(f"\nSampling posterior ({n_samples} samples)...")
    samples = BP.sample_posterior(n_samples)
    
    # Posterior statistics
    samples_array = np.asarray(samples.samples)
    posterior_mean = np.asarray(samples.mean())
    posterior_std = np.std(samples_array, axis=1)
    
    # 95% credible intervals
    lower_ci = np.percentile(samples_array, 2.5, axis=1)
    upper_ci = np.percentile(samples_array, 97.5, axis=1)
    
    # ========== LMRF Prior ==========
    print("\n--- LMRF prior (Laplace, sparsity-promoting) ---")
    x_map_lmrf = None
    try:
        x2 = LMRF(0, lmrf_precision, geometry=A.domain_geometry, name="x")
        y2 = Gaussian(mean=A @ x2, cov=noise_std**2, name="y")
        BP_lmrf = BayesianProblem(y2, x2).set_data(y=y_data)
        
        x_map_lmrf = np.asarray(BP_lmrf.MAP())
        print(f"  LMRF MAP computed successfully.")
    except Exception as e:
        print(f"  LMRF MAP failed: {e}")
        x_map_lmrf = None
    
    return {
        'x_map_gmrf': x_map_gmrf,
        'x_map_lmrf': x_map_lmrf,
        'posterior_mean': posterior_mean,
        'posterior_std': posterior_std,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'samples': samples_array,
        'n_samples': n_samples,
        'gmrf_precision': gmrf_precision,
        'lmrf_precision': lmrf_precision
    }
