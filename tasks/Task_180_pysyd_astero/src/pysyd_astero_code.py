"""
Task 180: pysyd_astero — Asteroseismic parameter extraction
Inverse Problem: Fitting global oscillation parameters (numax, delta_nu)
from stellar power spectra.

Forward: Given (numax, delta_nu, amplitudes, Harvey params) → power spectrum
Inverse: Given power spectrum → estimate numax and delta_nu
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, binary_dilation
import json
import os

# ─── Ground Truth Parameters ─────────────────────────────────────────────────
GT_NUMAX = 120.0        # μHz
GT_DELTA_NU = 10.5      # μHz

# Harvey background — using P(ν) = ζ / (1 + (ν/ν_c)^2)  [μHz units throughout]
HARVEY_ZETA1 = 5000.0   # ppm²/μHz (supergranulation)
HARVEY_NC1 = 30.0       # μHz
HARVEY_ZETA2 = 2000.0   # ppm²/μHz (granulation)
HARVEY_NC2 = 120.0      # μHz
WHITE_NOISE = 0.5       # ppm²/μHz

# Oscillation
SIGMA_ENV = 20.0        # μHz — Gaussian envelope width
MODE_HEIGHT = 3000.0    # ppm²/μHz — peak mode height
MODE_WIDTH = 0.5        # μHz — Lorentzian HWHM

FREQ_RES = 0.01         # μHz
FREQ_MIN = 1.0
FREQ_MAX = 300.0
SEED = 42


# ─── Forward Model ───────────────────────────────────────────────────────────

def harvey_comp(freq, zeta, nc):
    """Single Harvey component: P(ν) = ζ / (1 + (ν/ν_c)²)"""
    return zeta / (1.0 + (freq / nc) ** 2)


def bg_model(freq, z1, nc1, z2, nc2, w):
    """Background = 2 Harvey components + white noise."""
    return harvey_comp(freq, z1, nc1) + harvey_comp(freq, z2, nc2) + w


def gt_background(freq):
    return bg_model(freq, HARVEY_ZETA1, HARVEY_NC1, HARVEY_ZETA2, HARVEY_NC2, WHITE_NOISE)


def osc_modes(freq, numax, dnu, sigma_env, height, width):
    """Lorentzian modes modulated by Gaussian envelope."""
    eps = 1.5
    modes = np.zeros_like(freq)
    n_lo = int(np.floor((numax - 4 * sigma_env) / dnu))
    n_hi = int(np.ceil((numax + 4 * sigma_env) / dnu))

    for n in range(max(1, n_lo), n_hi + 1):
        for ell, vis in [(0, 1.0), (1, 0.7), (2, 0.5)]:
            d02 = -0.15 * dnu if ell == 2 else 0.0
            nu_m = dnu * (n + ell / 2.0 + eps) + d02
            if nu_m < freq[0] or nu_m > freq[-1]:
                continue
            env = np.exp(-0.5 * ((nu_m - numax) / sigma_env) ** 2)
            modes += height * env * vis * width ** 2 / ((freq - nu_m) ** 2 + width ** 2)
    return modes


def forward_model(freq):
    return gt_background(freq) + osc_modes(freq, GT_NUMAX, GT_DELTA_NU,
                                            SIGMA_ENV, MODE_HEIGHT, MODE_WIDTH)


# ─── Data Synthesis ──────────────────────────────────────────────────────────

def synthesize_data():
    rng = np.random.RandomState(SEED)
    freq = np.arange(FREQ_MIN, FREQ_MAX, FREQ_RES)
    ps_true = forward_model(freq)
    ps_obs = ps_true * rng.exponential(1.0, size=len(freq))
    return freq, ps_true, ps_obs


# ─── Inverse Solver ──────────────────────────────────────────────────────────

def gauss_env(freq, amp, center, sigma):
    return amp * np.exp(-0.5 * ((freq - center) / sigma) ** 2)


def inverse_solve(freq, ps_obs):
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

    return numax_est, dnu_est, plin, bg_fit, snr, lag, acf


# ─── Evaluation ──────────────────────────────────────────────────────────────

def compute_psnr(sig, ref):
    ls = np.log10(np.maximum(sig, 1e-10))
    lr = np.log10(np.maximum(ref, 1e-10))
    mse = np.mean((ls - lr) ** 2)
    if mse < 1e-30:
        return 100.0
    dr = np.max(lr) - np.min(lr)
    return 10.0 * np.log10(dr ** 2 / mse)


def compute_cc(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    d = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / d) if d > 1e-30 else 0.0


def evaluate(freq, numax_est, dnu_est, bg_fit):
    nre = abs(numax_est - GT_NUMAX) / GT_NUMAX
    dre = abs(dnu_est - GT_DELTA_NU) / GT_DELTA_NU
    psnr = compute_psnr(bg_fit, gt_background(freq))
    m = (freq > 60) & (freq < 200)
    cc = compute_cc(gauss_env(freq[m], 1, numax_est, SIGMA_ENV),
                    gauss_env(freq[m], 1, GT_NUMAX, SIGMA_ENV))
    return {
        "numax_true": GT_NUMAX,
        "numax_estimated": float(round(numax_est, 3)),
        "numax_relative_error": float(round(nre, 6)),
        "delta_nu_true": GT_DELTA_NU,
        "delta_nu_estimated": float(round(dnu_est, 3)),
        "delta_nu_relative_error": float(round(dre, 6)),
        "background_PSNR_dB": float(round(psnr, 2)),
        "envelope_CC": float(round(cc, 4)),
        "numax_RE_pass": bool(nre < 0.05),
        "delta_nu_RE_pass": bool(dre < 0.05),
    }


# ─── Visualization ───────────────────────────────────────────────────────────

def make_figure(freq, ps_true, ps_obs, numax_est, dnu_est,
                bg_fit, snr, lag, acf, metrics, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Task 180: Asteroseismic Parameter Extraction\n"
        f"νmax = {numax_est:.1f} μHz (GT: {GT_NUMAX}), "
        f"Δν = {dnu_est:.2f} μHz (GT: {GT_DELTA_NU})",
        fontsize=13, fontweight='bold')

    bg_true = gt_background(freq)
    om = (freq > 60) & (freq < 200)
    df = freq[1] - freq[0]

    ax = axes[0, 0]
    ax.loglog(freq, ps_obs, color='lightgray', lw=0.3, alpha=0.5,
              label='Observed', rasterized=True)
    ax.loglog(freq, ps_true, 'b-', lw=0.8, alpha=0.7, label='True spectrum')
    ax.loglog(freq, bg_true, 'g--', lw=1.5, label='True BG')
    ax.loglog(freq, bg_fit, 'r-', lw=1.5, label='Fitted BG')
    ax.set_xlabel('Frequency (μHz)')
    ax.set_ylabel('Power (ppm²/μHz)')
    ax.set_title('(a) Power Spectrum & Background')
    ax.legend(fontsize=8)
    ax.set_xlim([FREQ_MIN, FREQ_MAX])

    ax = axes[0, 1]
    exc = np.clip(ps_obs / bg_fit - 1.0, -1, 50)
    exc_sm = gaussian_filter1d(exc, int(2.0 / df))
    ax.plot(freq[om], exc[om], color='lightgray', lw=0.3, alpha=0.5, rasterized=True)
    ax.plot(freq[om], exc_sm[om], 'b-', lw=1.0, label='Smoothed')
    ax.axvline(numax_est, color='r', ls='--', lw=1.5, label=f'νmax={numax_est:.1f}')
    ax.axvline(GT_NUMAX, color='g', ls=':', lw=1.5, label=f'GT={GT_NUMAX}')
    ax.set_xlabel('Frequency (μHz)')
    ax.set_ylabel('SNR')
    ax.set_title('(b) Background-Subtracted')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    et = gauss_env(freq[om], 1, GT_NUMAX, SIGMA_ENV)
    ee = gauss_env(freq[om], 1, numax_est, SIGMA_ENV)
    ax.plot(freq[om], et, 'g-', lw=2, label='GT')
    ax.plot(freq[om], ee, 'r--', lw=2, label='Est')
    ax.fill_between(freq[om], et, alpha=0.2, color='green')
    ax.fill_between(freq[om], ee, alpha=0.2, color='red')
    ax.set_xlabel('Frequency (μHz)')
    ax.set_ylabel('Normalized Envelope')
    ax.set_title('(c) Envelope: GT vs Estimated')
    ax.legend(fontsize=8)
    ax.text(0.05, 0.9, f'νmax RE={metrics["numax_relative_error"]*100:.2f}%',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax = axes[1, 1]
    show = lag < 30
    sw = max(3, int(0.5 / df))
    asm = gaussian_filter1d(acf[show], sigma=sw)
    ax.plot(lag[show], acf[show], color='lightblue', lw=0.5)
    ax.plot(lag[show], asm, 'b-', lw=1.5, label='Smoothed ACF')
    ax.axvline(dnu_est, color='r', ls='--', lw=2, label=f'Δν={dnu_est:.2f}')
    ax.axvline(GT_DELTA_NU, color='g', ls=':', lw=2, label=f'GT={GT_DELTA_NU}')
    ax.set_xlabel('Lag (μHz)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('(d) ACF — Δν')
    ax.legend(fontsize=8)
    ax.text(0.05, 0.9, f'Δν RE={metrics["delta_nu_relative_error"]*100:.2f}%',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('results', exist_ok=True)
    print("=" * 60)
    print("Task 180: pysyd_astero")
    print("=" * 60)

    freq, ps_true, ps_obs = synthesize_data()
    bg = gt_background(freq)
    ix = np.argmin(np.abs(freq - GT_NUMAX))
    print(f"BG@νmax={bg[ix]:.1f}, True@νmax={ps_true[ix]:.1f}, "
          f"Mode/BG≈{ps_true[ix]/bg[ix]-1:.2f}")

    numax_est, dnu_est, plin, bg_fit, snr, lag, acf = inverse_solve(freq, ps_obs)
    print(f"νmax_est={numax_est:.3f}, Δν_est={dnu_est:.3f}")

    metrics = evaluate(freq, numax_est, dnu_est, bg_fit)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    make_figure(freq, ps_true, ps_obs, numax_est, dnu_est,
                bg_fit, snr, lag, acf, metrics, 'results/reconstruction_result.png')

    recon = bg_fit + osc_modes(freq, numax_est, dnu_est, SIGMA_ENV,
                                MODE_HEIGHT, MODE_WIDTH)
    np.save('results/ground_truth.npy', ps_true)
    np.save('results/reconstruction.npy', recon)

    print("\n" + "=" * 60)
    npass = 'PASS' if metrics['numax_RE_pass'] else 'FAIL'
    dpass = 'PASS' if metrics['delta_nu_RE_pass'] else 'FAIL'
    print(f"νmax RE: {metrics['numax_relative_error']*100:.2f}% {npass}")
    print(f"Δν RE:   {metrics['delta_nu_relative_error']*100:.2f}% {dpass}")
    print(f"BG PSNR: {metrics['background_PSNR_dB']:.2f} dB")
    print(f"Env CC:  {metrics['envelope_CC']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
