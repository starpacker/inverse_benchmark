import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import curve_fit

from scipy.signal import find_peaks

from scipy.ndimage import gaussian_filter1d, binary_dilation

def harvey_comp(freq, zeta, nc):
    """Single Harvey component: P(ν) = ζ / (1 + (ν/ν_c)²)"""
    return zeta / (1.0 + (freq / nc) ** 2)

def bg_model(freq, z1, nc1, z2, nc2, w):
    """Background = 2 Harvey components + white noise."""
    return harvey_comp(freq, z1, nc1) + harvey_comp(freq, z2, nc2) + w

def gauss_env(freq, amp, center, sigma):
    """Gaussian envelope function."""
    return amp * np.exp(-0.5 * ((freq - center) / sigma) ** 2)

def run_inversion(freq, ps_obs, params):
    """
    Inverse solver: Given power spectrum → estimate numax and delta_nu.
    
    Args:
        freq: frequency array
        ps_obs: observed power spectrum
        params: dictionary with ground truth parameters for reference
        
    Returns:
        result: dictionary containing estimated parameters and intermediate data
    """
    df = freq[1] - freq[0]

    # ── smooth in log-space ──
    log_ps = np.log10(np.maximum(ps_obs, 1e-10))
    ps_heavy = 10 ** gaussian_filter1d(log_ps, int(30.0 / df))   # ~30 μHz
    ps_med = 10 ** gaussian_filter1d(log_ps, int(3.0 / df))      # ~3 μHz

    # ── fit background (log-space), exclude central band initially ──
    def log_bg(f, lz1, lnc1, lz2, lnc2, lw):
        return np.log10(bg_model(f, 10**lz1, 10**lnc1, 10**lz2, 10**lnc2, 10**lw))

    p0 = [np.log10(5000), np.log10(30), np.log10(2000), np.log10(120), np.log10(0.5)]
    blo = [1, 0, 1, 0.5, -3]
    bhi = [6, 3, 6, 3, 3]

    mask0 = (freq < 50) | (freq > 250)
    try:
        popt, _ = curve_fit(log_bg, freq[mask0],
                            np.log10(np.maximum(ps_heavy[mask0], 1e-10)),
                            p0=p0, bounds=(blo, bhi), maxfev=30000)
    except Exception:
        popt = p0

    plin = [10**p for p in popt]
    bg_fit = bg_model(freq, *plin)

    # ── iterative: exclude oscillation bump, refit ──
    snr_r = ps_heavy / bg_fit
    bump = snr_r > 1.15
    bump_exp = binary_dilation(bump, structure=np.ones(int(15.0 / df)))
    mask1 = ~bump_exp
    if np.sum(mask1) > 200:
        try:
            popt2, _ = curve_fit(log_bg, freq[mask1],
                                 np.log10(np.maximum(ps_heavy[mask1], 1e-10)),
                                 p0=popt, bounds=(blo, bhi), maxfev=30000)
            plin = [10**p for p in popt2]
            bg_fit = bg_model(freq, *plin)
            popt = popt2
        except Exception:
            pass

    # ── SNR ──
    snr = ps_med / bg_fit - 1.0
    snr = np.maximum(snr, 0.0)

    # ── fit Gaussian → numax ──
    sel = (freq > 50) & (freq < 250)
    fs, ss = freq[sel], snr[sel]
    try:
        ip = np.argmax(ss)
        penv, _ = curve_fit(gauss_env, fs, ss,
                            p0=[ss[ip], fs[ip], 20.0],
                            bounds=([0, 50, 3], [1e6, 250, 80]), maxfev=30000)
        numax_est = penv[1]
        sig_est = abs(penv[2])
    except Exception:
        numax_est = fs[np.argmax(ss)]
        sig_est = 20.0

    # ── ACF → delta_nu ──
    half = max(2.5 * sig_est, 35.0)
    band = (freq > numax_est - half) & (freq < numax_est + half)
    excess = ps_obs[band] / bg_fit[band] - 1.0
    n = len(excess)
    ec = excess - np.mean(excess)
    fft_e = np.fft.rfft(ec, n=2 * n)
    acf_full = np.fft.irfft(np.abs(fft_e) ** 2)[:n]
    acf = acf_full / (acf_full[0] + 1e-30)
    lag = np.arange(n) * df

    lo, hi = 5.0, 20.0
    lm = (lag >= lo) & (lag <= hi)
    if np.any(lm):
        ls, acs = lag[lm], acf[lm]
        sw = max(3, int(0.5 / df))
        asm = gaussian_filter1d(acs, sigma=sw)
        pks, _ = find_peaks(asm, height=0.005, distance=int(2.0 / df))
        if len(pks) > 0:
            best = pks[np.argmax(asm[pks])]
            dnu_est = ls[best]
        else:
            dnu_est = ls[np.argmax(asm)]
    else:
        dnu_est = 10.0

    # parabolic refinement
    pi_ = np.argmin(np.abs(lag - dnu_est))
    if 1 < pi_ < len(acf) - 1:
        y0, y1, y2 = acf[pi_ - 1], acf[pi_], acf[pi_ + 1]
        d = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(d) > 1e-10:
            dnu_est = lag[pi_] + (y0 - y2) / d * df

    result = {
        'numax_est': numax_est,
        'dnu_est': dnu_est,
        'plin': plin,
        'bg_fit': bg_fit,
        'snr': snr,
        'lag': lag,
        'acf': acf
    }
    
    return result
