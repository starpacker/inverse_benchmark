import time

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')

PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "logA"]

def camb_Dl_TT(H0, ombh2, omch2, ns, logA, lmax):
    """Compute D_l^TT [µK²] for l=0..lmax using CAMB."""
    import camb
    As = 1e-10 * np.exp(logA)
    p = camb.CAMBparams()
    p.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=0.054)
    p.InitPower.set_params(As=As, ns=ns, r=0)
    p.set_for_lmax(lmax, lens_potential_accuracy=0)
    p.WantTensors = False
    p.Accuracy.AccuracyBoost = 1.0
    p.Accuracy.lAccuracyBoost = 1.0
    res = camb.get_results(p)
    pw = res.get_cmb_power_spectra(p, CMB_unit='muK')
    return pw['total'][:lmax + 1, 0]

def forward_operator(theta, lmax):
    """
    Forward model: compute CMB TT power spectrum from cosmological parameters.
    
    Parameters
    ----------
    theta : array-like
        Cosmological parameters [H0, ombh2, omch2, ns, logA]
    lmax : int
        Maximum multipole
    
    Returns
    -------
    ndarray
        D_l^TT power spectrum [µK²] for l=0..lmax
    """
    H0, ombh2, omch2, ns, logA = theta
    return camb_Dl_TT(H0, ombh2, omch2, ns, logA, lmax)

def run_inversion(data, true_params, prior_lo, prior_hi, n_samples, burn_in, seed=42):
    """
    Run MCMC sampling to estimate cosmological parameters.
    
    Parameters
    ----------
    data : dict
        Output from load_and_preprocess_data
    true_params : dict
        True cosmological parameters (used for initialization)
    prior_lo : ndarray
        Lower bounds of uniform prior
    prior_hi : ndarray
        Upper bounds of uniform prior
    n_samples : int
        Total number of MCMC samples
    burn_in : int
        Number of burn-in samples to discard
    seed : int
        Random seed
    
    Returns
    -------
    dict
        Dictionary containing:
        - parameter_results: dict with statistics for each parameter
        - posterior_samples: array of post-burn-in samples
        - Dl_recon: reconstructed power spectrum from median parameters
        - runtime: MCMC runtime in seconds
    """
    print(f"[2/5] MCMC sampling ({n_samples} steps, burn-in={burn_in}) ...")
    
    Dl_obs = data['Dl_obs']
    sigma = data['sigma']
    mask = data['mask']
    lmax = data['lmax']
    
    obs = Dl_obs[mask]
    ivar = 1.0 / sigma[mask]**2
    
    prop_std = np.array([0.20, 0.00008, 0.0008, 0.0025, 0.008])
    
    def logpost(theta):
        if np.any(theta < prior_lo) or np.any(theta > prior_hi):
            return -np.inf
        try:
            Dl = forward_operator(theta, lmax)
            return -0.5 * np.sum((obs - Dl[mask])**2 * ivar)
        except Exception:
            return -np.inf
    
    np.random.seed(seed)
    true_vec = np.array([true_params[p] for p in PARAM_NAMES])
    cur = true_vec + np.random.normal(0, prop_std * 0.3)
    cur_lp = logpost(cur)
    
    chain = np.zeros((n_samples, 5))
    lp_chain = np.zeros(n_samples)
    n_acc = 0
    t0 = time.time()
    
    for i in range(n_samples):
        prop = cur + np.random.normal(0, prop_std)
        plp = logpost(prop)
        if plp - cur_lp > np.log(np.random.uniform()):
            cur, cur_lp = prop, plp
            n_acc += 1
        chain[i] = cur
        lp_chain[i] = cur_lp
        
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"      Step {i + 1}/{n_samples}: {(i + 1) / el:.1f} it/s, "
                  f"accept={n_acc / (i + 1) * 100:.0f}%, logL={cur_lp:.1f}")
    
    elapsed = time.time() - t0
    print(f"      Done: {elapsed:.1f}s, accept={n_acc / n_samples * 100:.1f}%")
    
    post = chain[burn_in:]
    
    parameter_results = {}
    for j, pn in enumerate(PARAM_NAMES):
        s = post[:, j]
        parameter_results[pn] = {
            'true': true_params[pn],
            'median': float(np.median(s)),
            'mean': float(np.mean(s)),
            'std': float(np.std(s)),
            'ci16': float(np.percentile(s, 16)),
            'ci84': float(np.percentile(s, 84))
        }
    
    print("[3/5] Computing best-fit power spectrum ...")
    median_params = np.array([parameter_results[p]['median'] for p in PARAM_NAMES])
    Dl_recon = forward_operator(median_params, lmax)
    
    return {
        'parameter_results': parameter_results,
        'posterior_samples': post,
        'Dl_recon': Dl_recon,
        'runtime': elapsed
    }
