#!/usr/bin/env python3
"""
Rietveld Refinement from Powder X-ray Diffraction (PXRD)

Inverse Problem: Given a noisy powder diffraction pattern, refine crystal
structure parameters (lattice parameters, scale factor, profile parameters,
background) via weighted least-squares (Rietveld method).

Forward Model:
  - Peak positions from Bragg's law (cubic lattice parameter a)
  - Peak intensities: multiplicity × |F(hkl)|² × LP × Debye-Waller × scale
  - Peak shapes: pseudo-Voigt (Caglioti U,V,W for FWHM, η mixing)
  - Background: Chebyshev polynomial
  - Integrated intensities normalized so max peak ~ 10000 counts

Test System: CeO2 (Ceria), Fm-3m, a = 5.4116 Å, Cu Kα radiation
"""

import os
import json
import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

WAVELENGTH = 1.5406
TT_MIN, TT_MAX = 20.0, 130.0
NPTS = 2000
TRUE_A = 5.4116


# ─────────────────────────────────────────────
# Reflection table for CeO2 Fm-3m
# ─────────────────────────────────────────────
def _build_reflections():
    f_Ce, f_O = 58.0, 8.0
    refs = []
    for h in range(11):
        for k in range(h + 1):
            for ll in range(k + 1):
                if h == k == ll == 0:
                    continue
                if not (h % 2 == k % 2 == ll % 2):
                    continue
                fcc = [0, np.pi*(h+k), np.pi*(h+ll), np.pi*(k+ll)]
                fcc_sum = sum(np.exp(1j*p) for p in fcc)
                p1 = 2*np.pi*(h+k+ll)*0.25
                p2 = 2*np.pi*(h+k+ll)*0.75
                F = f_Ce*fcc_sum + f_O*fcc_sum*(np.exp(1j*p1) + np.exp(1j*p2))
                F2 = float(np.abs(F)**2)
                if F2 < 1:
                    continue
                if h == k == ll:
                    mult = 8
                elif k == 0 and ll == 0:
                    mult = 6
                elif h == k and ll == 0:
                    mult = 12
                elif h == k or k == ll:
                    mult = 24
                elif ll == 0:
                    mult = 24
                else:
                    mult = 48
                q2 = h*h + k*k + ll*ll
                refs.append((q2, F2, mult))
    return refs

REFS = _build_reflections()


# ─────────────────────────────────────────────
# Compute a normalization factor so peaks have
# reasonable height (~10000 max counts)
# ─────────────────────────────────────────────
def _compute_norm():
    """Find raw integrated intensity of strongest peak for normalization."""
    max_raw = 0
    for q2, F2, mult in REFS:
        d = TRUE_A / np.sqrt(q2)
        s = WAVELENGTH / (2*d)
        if abs(s) >= 1:
            continue
        tt = 2*np.degrees(np.arcsin(s))
        if tt < TT_MIN or tt > TT_MAX:
            continue
        th = np.radians(tt/2)
        lp = (1 + np.cos(2*th)**2) / (np.sin(th)**2 * np.cos(th))
        dw = np.exp(-0.5*(np.sin(th)/WAVELENGTH)**2)
        raw = mult * F2 * lp * dw
        if raw > max_raw:
            max_raw = raw
    return max_raw

RAW_NORM = _compute_norm()


# ─────────────────────────────────────────────
# Forward model (vectorized, fast)
# ─────────────────────────────────────────────
def compute_pattern(tt, p):
    """
    p = [a, scale, U, V, W, eta, bg0, bg1, bg2, bg3]
    scale ~ 1.0 produces max peak ~ 10000 counts
    """
    a, scale, U, V, W, eta = p[0], p[1], max(p[2], 1e-4), p[3], max(p[4], 1e-4), np.clip(p[5], 0.01, 0.99)
    bg = p[6:]

    # Chebyshev background
    x = 2*(tt - TT_MIN)/(TT_MAX - TT_MIN) - 1
    y = bg[0]*np.ones_like(x)
    if len(bg) > 1: y = y + bg[1]*x
    if len(bg) > 2: y = y + bg[2]*(2*x*x - 1)
    if len(bg) > 3: y = y + bg[3]*(4*x*x*x - 3*x)

    target_max = 10000.0  # normalize so scale=1 → max peak ~ target_max

    for q2, F2, mult in REFS:
        d = a / np.sqrt(q2)
        s = WAVELENGTH / (2*d)
        if abs(s) >= 1:
            continue
        tt_pk = 2*np.degrees(np.arcsin(s))
        if tt_pk < TT_MIN - 0.5 or tt_pk > TT_MAX + 0.5:
            continue
        th = np.radians(tt_pk/2)
        lp = (1 + np.cos(2*th)**2) / (np.sin(th)**2 * np.cos(th))
        dw = np.exp(-0.5*(np.sin(th)/WAVELENGTH)**2)

        raw_int = mult * F2 * lp * dw
        intensity = scale * target_max * raw_int / RAW_NORM

        # FWHM from Caglioti
        H = np.sqrt(max(U*np.tan(th)**2 + V*np.tan(th) + W, 1e-6))

        # Pseudo-Voigt
        sig = H / (2*np.sqrt(2*np.log(2)))
        gam = H / 2
        dx = tt - tt_pk
        gauss = np.exp(-0.5*(dx/sig)**2) / (sig*np.sqrt(2*np.pi))
        lorentz = (gam/np.pi) / (dx*dx + gam*gam)
        y = y + intensity * (eta*lorentz + (1-eta)*gauss)

    return y


def residuals(p, tt, yobs, w):
    return w * (yobs - compute_pattern(tt, p))


def rwp_val(yobs, ycalc, w):
    return np.sqrt(np.sum(w*w*(yobs-ycalc)**2) / np.sum(w*w*yobs**2)) * 100


# ─────────────────────────────────────────────
def main():
    print("="*60)
    print(" Rietveld Refinement: CeO2 Powder XRD")
    print("="*60)
    np.random.seed(42)

    tt = np.linspace(TT_MIN, TT_MAX, NPTS)

    # True: [a, scale, U, V, W, eta, bg0..bg3]
    true_p = np.array([TRUE_A, 1.0, 0.020, -0.005, 0.012, 0.50,
                       50.0, 15.0, -5.0, 2.0])
    print(f"\nTrue: a={true_p[0]:.4f} Å  scale={true_p[1]:.2f}")
    print(f"  U={true_p[2]:.4f} V={true_p[3]:.4f} W={true_p[4]:.4f} η={true_p[5]:.2f}")
    print(f"  bg={list(true_p[6:])}")

    # Ground truth pattern
    y_true = compute_pattern(tt, true_p)
    peak_max = np.max(y_true)
    bg_level = np.median(y_true[y_true < np.percentile(y_true, 30)])
    print(f"  Peak max: {peak_max:.0f}, background median: {bg_level:.1f}")

    # Add Poisson noise + readout
    y_obs = np.random.poisson(np.maximum(y_true, 0).astype(int)).astype(float)
    y_obs += np.random.normal(0, 3.0, NPTS)
    y_obs = np.maximum(y_obs, 0.1)

    noise_std = np.std(y_obs - y_true)
    print(f"  Noise σ={noise_std:.1f}, peak SNR={peak_max/noise_std:.0f}")

    w = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))

    # Initial guess (perturbed)
    init_p = np.array([TRUE_A * 1.008, 0.80, 0.030, -0.001, 0.018, 0.40,
                       35.0, 5.0, 0.0, 0.0])
    print(f"\nInitial: a={init_p[0]:.4f} Å (err={abs(init_p[0]-TRUE_A)/TRUE_A*100:.2f}%)")
    print(f"  scale={init_p[1]:.2f} (err={abs(init_p[1]-true_p[1])/true_p[1]*100:.0f}%)")

    # Refinement
    print("\nRefining...")
    lb = [4.5, 0.01, 0.0001, -0.1, 0.0001, 0.01, -200, -200, -200, -200]
    ub = [6.5, 10.0, 0.5,    0.1,  0.5,    0.99, 200,  200,  200,  200]

    result = least_squares(residuals, init_p, args=(tt, y_obs, w),
                           bounds=(lb, ub), method='trf',
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           max_nfev=5000, verbose=1)

    rp = result.x
    y_calc = compute_pattern(tt, rp)

    # Metrics
    rwp = rwp_val(y_obs, y_calc, w)
    np_ = len(rp)
    rexp = np.sqrt((NPTS - np_) / np.sum(w*w*y_obs**2)) * 100
    gof = rwp / rexp
    chi2 = np.sum(w*w*(y_obs-y_calc)**2) / (NPTS - np_)
    lat_re = abs(rp[0]-true_p[0])/true_p[0]*100
    cc = float(np.corrcoef(y_obs, y_calc)[0, 1])
    mse = np.mean((y_true - y_calc)**2)
    psnr = 10*np.log10(peak_max**2/mse) if mse > 0 else 999.0
    scale_re = abs(rp[1]-true_p[1])/true_p[1]*100
    U_re = abs(rp[2]-true_p[2])/abs(true_p[2])*100
    W_re = abs(rp[4]-true_p[4])/abs(true_p[4])*100
    eta_re = abs(rp[5]-true_p[5])/abs(true_p[5])*100

    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    for name, true, ref, re_v in [
        ('a (Å)', true_p[0], rp[0], lat_re),
        ('scale', true_p[1], rp[1], scale_re),
        ('U', true_p[2], rp[2], U_re),
        ('W', true_p[4], rp[4], W_re),
        ('η', true_p[5], rp[5], eta_re),
    ]:
        print(f"  {name:8s}: {ref:10.5f}  (true {true:.4f}, RE {re_v:.3f}%)")
    print(f"  V       : {rp[3]:10.5f}  (true {true_p[3]:.4f})")
    print(f"  bg: {[f'{b:.2f}' for b in rp[6:]]}")
    print(f"  true bg: {list(true_p[6:])}")
    print(f"\n  Rwp={rwp:.3f}%  Rexp={rexp:.3f}%  GoF={gof:.3f}  χ²={chi2:.4f}")
    print(f"  CC={cc:.6f}  PSNR={psnr:.2f} dB  lat_RE={lat_re:.4f}%")

    # Save metrics
    n_peaks = sum(
        1 for q2, _, _ in REFS
        if (ss := WAVELENGTH/(2*true_p[0]/np.sqrt(q2))) < 1
        and TT_MIN <= 2*np.degrees(np.arcsin(ss)) <= TT_MAX
    )
    metrics = {
        "task_name": "gsas2_rietveld",
        "method": "Pure-Python Rietveld (least-squares, pseudo-Voigt, Caglioti FWHM)",
        "crystal_system": "CeO2, Fm-3m, fluorite",
        "wavelength_angstrom": WAVELENGTH,
        "true_lattice_a": float(true_p[0]),
        "refined_lattice_a": round(float(rp[0]), 5),
        "lattice_param_RE_percent": round(lat_re, 4),
        "Rwp_percent": round(rwp, 3),
        "Rexp_percent": round(rexp, 3),
        "GoF": round(gof, 3),
        "chi_squared_reduced": round(chi2, 4),
        "pattern_CC": round(cc, 6),
        "PSNR_dB": round(psnr, 2),
        "scale_true": float(true_p[1]),
        "scale_refined": round(float(rp[1]), 4),
        "scale_RE_percent": round(scale_re, 2),
        "U": round(float(rp[2]), 5), "V": round(float(rp[3]), 5),
        "W": round(float(rp[4]), 5), "eta": round(float(rp[5]), 4),
        "num_peaks": n_peaks,
        "two_theta_range_deg": f"{TT_MIN}-{TT_MAX}",
        "num_points": NPTS,
        "converged": bool(result.success),
        "n_evals": int(result.nfev)
    }
    mp = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(mp, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {mp}")

    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'),
            {'two_theta': tt, 'y_true': y_true, 'y_obs': y_obs,
             'true_params': true_p}, allow_pickle=True)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'),
            {'two_theta': tt, 'y_calc': y_calc,
             'refined_params': rp}, allow_pickle=True)
    print("Saved ground_truth.npy, reconstruction.npy")

    # ── Visualization ──
    print("Generating visualization...")
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.08, 1.3], hspace=0.32)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(tt, y_obs, 'b-', lw=0.3, alpha=0.5, label='Observed')
    ax1.plot(tt, y_calc, 'r-', lw=0.8, alpha=0.85, label='Calculated (Rietveld)')
    for q2, _, _ in REFS:
        d = rp[0]/np.sqrt(q2)
        s = WAVELENGTH/(2*d)
        if abs(s) < 1:
            tp = 2*np.degrees(np.arcsin(s))
            if TT_MIN <= tp <= TT_MAX:
                ax1.axvline(tp, ymin=0, ymax=0.02, color='green', lw=0.6, alpha=0.5)
    ax1.set_ylabel('Intensity (counts)', fontsize=12)
    ax1.set_title(f'Rietveld Fit: CeO2 (Fm-3m, Cu Kα)  |  Rwp={rwp:.2f}%  '
                  f'CC={cc:.5f}  GoF={gof:.2f}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim(TT_MIN, TT_MAX)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    diff = y_obs - y_calc
    ax2.plot(tt, diff, 'g-', lw=0.4)
    ax2.axhline(0, color='k', lw=0.5, ls='--')
    ax2.fill_between(tt, diff, 0, alpha=0.1, color='green')
    ax2.set_ylabel('Δ', fontsize=11)
    ax2.set_xlabel('2θ (degrees)', fontsize=12)
    ax2.set_title('Difference (Obs − Calc)', fontsize=10)
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')
    rows = [
        ['Param', 'True', 'Initial', 'Refined', 'RE(%)'],
        ['a (Å)', f'{true_p[0]:.4f}', f'{init_p[0]:.4f}', f'{rp[0]:.5f}', f'{lat_re:.4f}'],
        ['Scale', f'{true_p[1]:.2f}', f'{init_p[1]:.2f}', f'{rp[1]:.4f}', f'{scale_re:.2f}'],
        ['U', f'{true_p[2]:.4f}', f'{init_p[2]:.4f}', f'{rp[2]:.5f}', f'{U_re:.2f}'],
        ['W', f'{true_p[4]:.4f}', f'{init_p[4]:.4f}', f'{rp[4]:.5f}', f'{W_re:.2f}'],
        ['η', f'{true_p[5]:.2f}', f'{init_p[5]:.2f}', f'{rp[5]:.4f}', f'{eta_re:.2f}'],
    ]
    tbl = ax3.table(cellText=rows, loc='center', cellLoc='center',
                    colWidths=[0.12, 0.14, 0.14, 0.16, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)
    for j in range(5):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        try:
            v = float(rows[i][4])
            tbl[i, 4].set_facecolor('#C6EFCE' if v < 1 else '#FFEB9C' if v < 5 else '#FFC7CE')
        except ValueError:
            pass
    ax3.set_title('Refined Parameters', fontsize=11, pad=8)

    fig.text(0.5, 0.005,
             f'Rwp={rwp:.2f}%  |  GoF={gof:.2f}  |  χ²={chi2:.2f}  |  '
             f'a_RE={lat_re:.4f}%  |  CC={cc:.5f}  |  PSNR={psnr:.1f} dB',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    vp = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
    plt.savefig(vp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {vp}")

    print(f"\n{'='*55}")
    print(" PIPELINE COMPLETE")
    print(f"{'='*55}")
    return metrics


if __name__ == '__main__':
    main()
