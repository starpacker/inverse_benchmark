import matplotlib

matplotlib.use('Agg')

import numpy as np

from numpy.fft import fft, ifft

def run_inversion(measured_spectrum, zlp, energy_axis, taper_threshold=0.02):
    """
    Fourier-log deconvolution with frequency-domain tapering.
    
    Algorithm:
        1. Compute ratio R(q) = F[J](q) / F[Z](q) with regularization.
        2. Apply smooth taper T(q) that rolls off for |F[Z](q)| < threshold.
        3. Recovered SSD = F^{-1}[ T(q) * ln(R(q)) ]
    
    The taper prevents noise amplification at frequencies where the ZLP's
    Fourier transform is vanishingly small.
    
    Parameters
    ----------
    measured_spectrum : ndarray
        Measured EELS spectrum.
    zlp : ndarray
        Zero-loss peak.
    energy_axis : ndarray
        Energy axis in eV.
    taper_threshold : float
        Frequencies with |F[Z]| / max(|F[Z]|) below this are tapered to zero.
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'ssd_recovered': Recovered SSD = (t/lambda) * S_norm
        - 'energy_axis': Energy axis
    """
    J = fft(measured_spectrum)
    Z = fft(zlp)
    
    # Frequency-domain taper: smooth rolloff where |Z| is small
    Z_abs_norm = np.abs(Z) / np.max(np.abs(Z))
    taper = np.clip(Z_abs_norm / taper_threshold, 0.0, 1.0)
    
    # Regularized division
    Z_reg = Z + 1e-10 * np.max(np.abs(Z))
    ratio = J / Z_reg
    
    # Complex logarithm with taper
    log_ratio = (np.log(np.abs(ratio) + 1e-30) + 1j * np.angle(ratio)) * taper
    
    ssd_recovered = np.real(ifft(log_ratio))
    
    # Enforce causality: zero outside physical region
    ssd_recovered[energy_axis < 2.0] = 0.0
    ssd_recovered[energy_axis > 100.0] = 0.0
    
    result = {
        'ssd_recovered': ssd_recovered,
        'energy_axis': energy_axis
    }
    
    return result
