import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')

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
