import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import cho_factor, cho_solve

import emcee

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def powerlaw_psd(freqs, log10_A, gamma):
    """Power-law power spectral density: S(f) = A^2/(12*pi^2) * (f/f_yr)^(-gamma) * f_yr^(-3)."""
    A = 10.0 ** log10_A
    f_yr = 1.0 / (365.25 * 86400.0)
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs / f_yr) ** (-gamma) * f_yr ** (-3)

def forward_operator(params, data_dict):
    """
    Compute the predicted power spectral densities given model parameters.
    
    This is the forward model that maps parameters to observables (PSDs).
    
    Args:
        params: Array [log10_A_gw, log10_A_red, gamma_red]
        data_dict: Dictionary containing freqs and other data
        
    Returns:
        psd_dict: Dictionary containing predicted red noise and GW PSDs
    """
    log10_A_gw, log10_A_red, gamma_red = params
    freqs = data_dict['freqs']
    
    # Compute predicted PSDs
    psd_red_pred = powerlaw_psd(freqs, log10_A_red, gamma_red)
    psd_gw_pred = powerlaw_psd(freqs, log10_A_gw, 13.0 / 3.0)
    
    return {
        'psd_red': psd_red_pred,
        'psd_gw': psd_gw_pred,
        'freqs': freqs
    }

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

def _log_prior(params):
    """Uniform priors on parameters."""
    log10_A_gw, log10_A_red, gamma_red = params
    if not (-18.0 < log10_A_gw < -11.0):
        return -np.inf
    if not (-18.0 < log10_A_red < -11.0):
        return -np.inf
    if not (0.0 < gamma_red < 10.0):
        return -np.inf
    return 0.0

def _log_posterior(params, data_dict):
    """Log-posterior = log-prior + log-likelihood."""
    lp = _log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = _log_likelihood(params, data_dict)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_inversion(data_dict, n_walkers, n_steps, n_burn):
    """
    Run MCMC inversion to recover PTA parameters.
    
    Args:
        data_dict: Dictionary containing all preprocessed data
        n_walkers: Number of MCMC walkers
        n_steps: Number of MCMC steps
        n_burn: Number of burn-in steps to discard
        
    Returns:
        result_dict: Dictionary containing recovered parameters and samples
    """
    true_params = data_dict['true_params']
    ndim = 3

    # Initialise walkers near truth (with scatter)
    p0 = true_params + 0.3 * np.random.randn(n_walkers, ndim)
    # Clip to prior bounds
    p0[:, 0] = np.clip(p0[:, 0], -17.9, -11.1)
    p0[:, 1] = np.clip(p0[:, 1], -17.9, -11.1)
    p0[:, 2] = np.clip(p0[:, 2], 0.1, 9.9)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _log_posterior,
        args=(data_dict,)
    )
    sampler.run_mcmc(p0, n_steps, progress=True)

    # Extract samples after burn-in
    samples = sampler.get_chain(discard=n_burn, flat=True)
    chain = sampler.get_chain()
    
    # Compute posterior statistics
    medians = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    # Compute predicted PSDs using recovered parameters
    psd_dict = forward_operator(medians, data_dict)
    
    result_dict = {
        'samples': samples,
        'chain': chain,
        'medians': medians,
        'stds': stds,
        'psd_red_recon': psd_dict['psd_red'],
        'psd_gw_recon': psd_dict['psd_gw'],
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'n_burn': n_burn,
    }
    
    return result_dict
