import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import cho_factor, cho_solve

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def powerlaw_psd(freqs, log10_A, gamma):
    """Power-law power spectral density: S(f) = A^2/(12*pi^2) * (f/f_yr)^(-gamma) * f_yr^(-3)."""
    A = 10.0 ** log10_A
    f_yr = 1.0 / (365.25 * 86400.0)
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs / f_yr) ** (-gamma) * f_yr ** (-3)

def _log_likelihood(params, data_dict):
    """Marginalised log-likelihood for PTA (Fourier-domain)."""
    log10_A_gw, log10_A_red, gamma_red = params
    
    toas_all = data_dict['toas_all']
    residuals_all = data_dict['residuals_all']
    F_all = data_dict['F_all']
    freqs = data_dict['freqs']
    hd_matrix = data_dict['hd_matrix']
    t_span = data_dict['t_span']
    n_pulsars = data_dict['n_pulsars']
    n_toa = data_dict['n_toa']
    n_freq = data_dict['n_freq']
    sigma_wn = data_dict['sigma_wn']
    efac = data_dict['efac']

    psd_red = powerlaw_psd(freqs, log10_A_red, gamma_red)
    psd_gw = powerlaw_psd(freqs, log10_A_gw, 13.0 / 3.0)

    logL = 0.0
    sigma_wn_sq = (sigma_wn * efac) ** 2

    for p in range(n_pulsars):
        F = F_all[p]
        r = residuals_all[p]

        # Per-pulsar phi: red + GW (diagonal part for this pulsar)
        phi_diag = np.zeros(2 * n_freq)
        for k in range(n_freq):
            phi_val = (psd_red[k] + hd_matrix[p, p] * psd_gw[k]) * t_span
            phi_diag[2 * k] = phi_val
            phi_diag[2 * k + 1] = phi_val

        # Woodbury identity for inversion
        phi_inv = np.where(phi_diag > 0, 1.0 / phi_diag, 1e30)
        FNr = F.T @ (r / sigma_wn_sq)
        FNF = F.T @ F / sigma_wn_sq
        Sigma_a = FNF + np.diag(phi_inv)

        try:
            cf = cho_factor(Sigma_a)
            Sigma_a_inv_FNr = cho_solve(cf, FNr)
            log_det_Sigma_a = 2.0 * np.sum(np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            return -1e100

        # log-likelihood contribution
        rNr = np.sum(r ** 2) / sigma_wn_sq
        logL_p = -0.5 * rNr + 0.5 * FNr @ Sigma_a_inv_FNr
        logL_p -= 0.5 * log_det_Sigma_a
        logL_p -= 0.5 * np.sum(np.log(phi_diag[phi_diag > 0]))

        logL += logL_p

    return logL
