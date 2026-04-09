import time

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')

NOISE_UK_ARCMIN = 45.0

BEAM_ARCMIN = 7.0

def compute_noise_Dl(lmax):
    """Compute white-noise + Gaussian-beam noise D_l."""
    ell = np.arange(lmax + 1, dtype=float)
    nr = NOISE_UK_ARCMIN * np.pi / (180 * 60)
    sb = BEAM_ARCMIN * np.pi / (180 * 60) / np.sqrt(8 * np.log(2))
    Nl = nr**2 * np.exp(ell * (ell + 1) * sb**2)
    Dl = np.zeros_like(ell)
    Dl[2:] = ell[2:] * (ell[2:] + 1) / (2 * np.pi) * Nl[2:]
    return Dl

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

def load_and_preprocess_data(true_params, lmin, lmax, fsky, seed=42):
    """
    Generate synthetic CMB TT power spectrum data.
    
    Parameters
    ----------
    true_params : dict
        True cosmological parameters (H0, ombh2, omch2, ns, logA)
    lmin : int
        Minimum multipole
    lmax : int
        Maximum multipole
    fsky : float
        Sky fraction observed
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - ells: multipole values
        - Dl_true: true power spectrum
        - Dl_obs: observed (noisy) power spectrum
        - sigma: uncertainty on each Dl
        - mask: boolean mask for valid multipoles
    """
    print("[1/5] Generating true CMB TT power spectrum ...")
    t0 = time.time()
    
    Dl_true = camb_Dl_TT(
        H0=true_params['H0'],
        ombh2=true_params['ombh2'],
        omch2=true_params['omch2'],
        ns=true_params['ns'],
        logA=true_params['logA'],
        lmax=lmax
    )
    print(f"      CAMB: {time.time() - t0:.2f}s, lmax={lmax}")
    
    ells = np.arange(len(Dl_true), dtype=float)
    Dl_n = compute_noise_Dl(lmax)
    
    sigma = np.zeros_like(Dl_true)
    for l in range(lmin, len(Dl_true)):
        fac = 2 * np.pi / (l * (l + 1))
        Cl_s = Dl_true[l] * fac
        Cl_n = Dl_n[l] * fac
        sig_Cl = np.sqrt(2 / ((2 * l + 1) * fsky)) * (Cl_s + Cl_n)
        sigma[l] = sig_Cl / fac
    
    np.random.seed(seed)
    Dl_obs = Dl_true.copy()
    for l in range(lmin, len(Dl_obs)):
        Dl_obs[l] += np.random.normal(0, sigma[l])
    
    mask = np.zeros(len(Dl_obs), dtype=bool)
    mask[lmin:lmax + 1] = True
    
    return {
        'ells': ells,
        'Dl_true': Dl_true,
        'Dl_obs': Dl_obs,
        'sigma': sigma,
        'mask': mask,
        'lmin': lmin,
        'lmax': lmax
    }
