"""
namaster_pseudo - Pseudo-Cl Power Spectrum Deconvolution
========================================================
Task: Estimate true angular power spectrum from masked sky observations
Repo: https://github.com/LSSTDESC/NaMaster

Forward problem:
    Given a true power spectrum Cl, generate a full-sky CMB map via
    healpy.synfast, then apply a partial-sky mask. The observed pseudo-Cl
    from the masked map is biased by mode-coupling.

Inverse problem:
    Use NaMaster (pymaster) to compute the mode-coupling matrix from the
    mask and decouple the pseudo-Cl to recover the true Cl.

Usage:
    /data/yjh/namaster_pseudo_env/bin/python namaster_pseudo_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import healpy as hp
import pymaster as nmt


def generate_data(nside=64, lmax=None):
    """
    Generate simulated CMB map with known Cl and a galactic mask.

    Returns
    -------
    cl_true : array, shape (lmax+1,)
        Input theoretical power spectrum.
    full_map : array, shape (npix,)
        Full-sky realization drawn from cl_true.
    mask_apo : array, shape (npix,)
        Apodized binary mask (galactic cut).
    nside : int
    lmax : int
    """
    if lmax is None:
        lmax = 2 * nside

    # ---- theoretical power spectrum (CMB-like) ----
    ell = np.arange(lmax + 1, dtype=float)
    cl_true = np.zeros(lmax + 1)
    cl_true[2:] = 1.0 / (ell[2:] * (ell[2:] + 1))
    cl_true *= 1e4  # scale for realistic amplitude

    # ---- full-sky realization ----
    np.random.seed(42)
    full_map = hp.synfast(cl_true, nside, lmax=lmax, verbose=False)

    # ---- galactic mask: cut |b| < 20° ----
    npix = hp.nside2npix(nside)
    mask = np.ones(npix)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    lat = np.pi / 2 - theta
    mask[np.abs(lat) < np.radians(20)] = 0

    # ---- apodize mask (C1 taper, 10° scale) ----
    mask_apo = nmt.mask_apodization(mask, 10.0, apotype='C1')

    return cl_true, full_map, mask_apo, nside, lmax


def compute_pseudo_cl(full_map, mask, nside, lmax):
    """
    Naive pseudo-Cl: anafast on masked map, divided by f_sky.
    This is the *biased* estimate (forward observation).
    """
    masked_map = full_map * mask
    cl_pseudo = hp.anafast(masked_map, lmax=lmax)
    f_sky = np.mean(mask ** 2)
    cl_pseudo_corrected = cl_pseudo / f_sky
    return cl_pseudo_corrected


def reconstruct_cl(full_map, mask, nside, lmax):
    """
    NaMaster deconvolution: compute the mode-coupling matrix from the
    mask and apply its inverse to obtain unbiased Cl estimates.

    Returns
    -------
    cl_decoupled : array – decoupled power spectrum in each bin
    ell_eff : array – effective multipole of each bin
    bins : NmtBin object
    """
    # NaMaster field: pass the *unmasked* map; NaMaster applies the mask
    f = nmt.NmtField(mask, [full_map])

    # Binning scheme – 4 multipoles per bin
    bin_size = 4
    b = nmt.NmtBin.from_nside_linear(nside, bin_size)

    # Workspace: compute the mode-coupling matrix
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)

    # Decouple: pseudo-Cl -> true Cl
    cl_decoupled = nmt.compute_full_master(f, f, b)

    ell_eff = b.get_effective_ells()
    return cl_decoupled[0], ell_eff, b


def compute_metrics(cl_true, cl_recon, ell_eff):
    """
    Evaluate reconstruction quality.

    Metrics
    -------
    PSNR (dB), Pearson correlation coefficient, RMSE, mean relative error.
    """
    # Interpolate true Cl at the effective ell values of each bin
    ell_all = np.arange(len(cl_true))
    cl_true_binned = np.interp(ell_eff, ell_all, cl_true)

    # Keep only ell >= 2 (monopole/dipole undefined)
    valid = ell_eff >= 2
    t = cl_true_binned[valid]
    r = cl_recon[valid]

    # PSNR
    data_range = np.max(t) - np.min(t)
    mse = np.mean((t - r) ** 2)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')

    # Pearson CC
    cc = float(np.corrcoef(t, r)[0, 1])

    # Relative error
    re = float(np.mean(np.abs(t - r) / (np.abs(t) + 1e-30)))

    # RMSE
    rmse = float(np.sqrt(mse))

    return {
        "psnr_dB": float(psnr),
        "correlation_coefficient": cc,
        "rmse": rmse,
        "mean_relative_error": re,
        "method": "NaMaster_pseudo_Cl_deconvolution",
    }


def visualize(cl_true, cl_pseudo, cl_recon, ell_eff, lmax, metrics, save_path):
    """Four-panel visualization of the deconvolution result."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ell_all = np.arange(len(cl_true))
    cl_true_at_ell = np.interp(ell_eff, ell_all, cl_true)

    # helper: D_ell = ell*(ell+1)*Cl / 2pi
    def D(ell_arr, cl_arr):
        return ell_arr * (ell_arr + 1) * cl_arr / (2 * np.pi)

    # ---- (a) True power spectrum ----
    ax = axes[0, 0]
    ax.plot(ell_all[2:], D(ell_all[2:], cl_true[2:]), 'b-', lw=1.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(a) True Power Spectrum')
    ax.set_xlim([2, lmax])

    # ---- (b) Pseudo-Cl (naive) vs True ----
    ax = axes[0, 1]
    ax.plot(ell_all[2:], D(ell_all[2:], cl_pseudo[2:]), 'r-', alpha=0.7, lw=1, label='Pseudo-Cℓ')
    ax.plot(ell_all[2:], D(ell_all[2:], cl_true[2:]), 'b--', alpha=0.5, lw=1, label='True')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(b) Pseudo-Cℓ (biased) vs True')
    ax.legend()
    ax.set_xlim([2, lmax])

    # ---- (c) Decoupled Cl (NaMaster) vs True ----
    ax = axes[1, 0]
    ax.plot(ell_eff, D(ell_eff, cl_recon), 'go-', ms=3, lw=1.5, label='NaMaster')
    ax.plot(ell_eff, D(ell_eff, cl_true_at_ell), 'b--', lw=1, label='True')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(c) Decoupled Cℓ (NaMaster) vs True')
    ax.legend()
    ax.set_xlim([2, lmax])

    # ---- (d) Relative error per bin ----
    ax = axes[1, 1]
    valid = ell_eff >= 2
    rel_err = np.abs(cl_recon[valid] - cl_true_at_ell[valid]) / (np.abs(cl_true_at_ell[valid]) + 1e-30)
    ax.semilogy(ell_eff[valid], rel_err, 'k.-', ms=3)
    ax.axhline(y=0.1, color='r', ls='--', alpha=0.5, label='10% error')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('|Cℓ_recon − Cℓ_true| / Cℓ_true')
    ax.set_title('(d) Relative Error per ℓ-bin')
    ax.legend()
    ax.set_xlim([2, lmax])

    fig.suptitle(
        f"NaMaster Pseudo-Cℓ Deconvolution  |  "
        f"PSNR={metrics['psnr_dB']:.2f} dB  |  "
        f"CC={metrics['correlation_coefficient']:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ======================================================================
#  Main pipeline
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  namaster_pseudo — Pseudo-Cℓ Deconvolution Pipeline")
    print("=" * 60)

    # (a) Generate data -------------------------------------------------
    cl_true, full_map, mask, nside, lmax = generate_data(nside=64)
    print(f"[DATA] nside={nside}, lmax={lmax}, npix={hp.nside2npix(nside)}")
    print(f"[DATA] f_sky = {np.mean(mask**2):.3f}")

    # (b) Naive pseudo-Cl (forward observation, biased) -----------------
    cl_pseudo = compute_pseudo_cl(full_map, mask, nside, lmax)
    print("[PSEUDO] Computed naive pseudo-Cℓ (f_sky-corrected)")

    # (c) NaMaster deconvolution (inverse) ------------------------------
    cl_recon, ell_eff, bins = reconstruct_cl(full_map, mask, nside, lmax)
    print(f"[RECON] Decoupled Cℓ: {len(ell_eff)} bins, "
          f"ℓ ∈ [{ell_eff[0]:.0f}, {ell_eff[-1]:.0f}]")

    # (d) Evaluate metrics ----------------------------------------------
    metrics = compute_metrics(cl_true, cl_recon, ell_eff)
    print(f"[EVAL] PSNR  = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC    = {metrics['correlation_coefficient']:.6f}")
    print(f"[EVAL] RMSE  = {metrics['rmse']:.6e}")
    print(f"[EVAL] RE    = {metrics['mean_relative_error']:.4f}")

    # (e) Save metrics --------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # (f) Visualize -----------------------------------------------------
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize(cl_true, cl_pseudo, cl_recon, ell_eff, lmax, metrics, vis_path)

    # (g) Save arrays ---------------------------------------------------
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), cl_true)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), cl_recon)
    np.save(os.path.join(RESULTS_DIR, "observed_data.npy"), cl_pseudo)
    np.save(os.path.join(RESULTS_DIR, "ell_effective.npy"), ell_eff)
    print("[SAVE] Arrays saved (ground_truth, recon_output, observed_data, ell_effective)")

    print("=" * 60)
    print("  DONE")
    print("=" * 60)
