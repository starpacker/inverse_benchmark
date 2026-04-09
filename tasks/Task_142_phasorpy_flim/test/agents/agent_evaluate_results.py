import json

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

from phasorpy.phasor import (
    phasor_from_lifetime,
    phasor_from_signal,
    phasor_semicircle,
)

def evaluate_results(
    f1_gt: np.ndarray,
    f1_recon: np.ndarray,
    tau1_ns: float,
    tau2_ns: float,
    freq_mhz: float,
    total_photons: int,
    nx: int,
    ny: int,
    G_meas: np.ndarray,
    S_meas: np.ndarray,
    G_ref: np.ndarray,
    S_ref: np.ndarray,
    outdir: str,
) -> dict:
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Computes PSNR, SSIM, MAE, and lifetime-related metrics.
    Saves metrics, arrays, and a 4-panel visualization figure.
    
    Parameters
    ----------
    f1_gt : np.ndarray
        Ground truth fraction map for species 1.
    f1_recon : np.ndarray
        Reconstructed fraction map for species 1.
    tau1_ns, tau2_ns : float
        Lifetimes of species 1 and 2 in nanoseconds.
    freq_mhz : float
        Laser repetition frequency in MHz.
    total_photons : int
        Mean total photon count per pixel.
    nx, ny : int
        Image dimensions.
    G_meas, S_meas : np.ndarray
        Measured phasor coordinates.
    G_ref, S_ref : np.ndarray
        Reference phasor coordinates for the two components.
    outdir : str
        Output directory for saving results.
    
    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    os.makedirs(outdir, exist_ok=True)
    
    f2_gt = 1.0 - f1_gt
    
    # Compute metrics
    psnr_val = psnr(f1_gt, f1_recon, data_range=1.0)
    ssim_val = ssim(f1_gt, f1_recon, data_range=1.0)
    
    # Lifetime-related relative errors
    tau_eff_gt = f1_gt * tau1_ns + f2_gt * tau2_ns
    tau_eff_recon = f1_recon * tau1_ns + (1 - f1_recon) * tau2_ns
    tau_re = np.mean(np.abs(tau_eff_gt - tau_eff_recon) / tau_eff_gt)
    
    # Mean absolute error of fraction
    mae_f1 = np.mean(np.abs(f1_gt - f1_recon))
    
    metrics = {
        "PSNR_dB": round(float(psnr_val), 2),
        "SSIM": round(float(ssim_val), 6),
        "fraction_MAE": round(float(mae_f1), 6),
        "lifetime_eff_RE": round(float(tau_re), 6),
        "tau1_ns": tau1_ns,
        "tau2_ns": tau2_ns,
        "frequency_MHz": freq_mhz,
        "image_size": [nx, ny],
        "total_photons_per_pixel": total_photons,
    }
    
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(outdir, "ground_truth.npy"), f1_gt)
    np.save(os.path.join(outdir, "recon_output.npy"), f1_recon)
    
    print("\nSaved ground_truth.npy, recon_output.npy, metrics.json")
    
    # Visualization: 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    # Panel 1: GT fraction map
    im0 = axes[0, 0].imshow(f1_gt, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[0, 0].set_title("Ground Truth: Species-1 Fraction", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], label="$f_1$")
    
    # Panel 2: Phasor plot
    ax_ph = axes[0, 1]
    # Draw universal semicircle
    sc_g, sc_s = phasor_semicircle()
    ax_ph.plot(sc_g, sc_s, "k-", linewidth=1.5, label="Universal semicircle")
    # Scatter all pixel phasors (subsample for clarity)
    step = max(1, nx * ny // 3000)
    g_flat = G_meas.ravel()[::step]
    s_flat = S_meas.ravel()[::step]
    f1_flat = f1_gt.ravel()[::step]
    sc = ax_ph.scatter(g_flat, s_flat, c=f1_flat, cmap="viridis", s=3, alpha=0.5,
                       vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_ph, label="$f_1$ (GT)")
    # Mark component positions
    ax_ph.plot(G_ref[0], S_ref[0], "r^", markersize=12, label=f"τ₁={tau1_ns} ns")
    ax_ph.plot(G_ref[1], S_ref[1], "bs", markersize=12, label=f"τ₂={tau2_ns} ns")
    ax_ph.set_xlabel("G (real)", fontsize=11)
    ax_ph.set_ylabel("S (imaginary)", fontsize=11)
    ax_ph.set_title("Phasor Plot", fontsize=12)
    ax_ph.set_xlim(-0.05, 1.05)
    ax_ph.set_ylim(-0.05, 0.6)
    ax_ph.set_aspect("equal")
    ax_ph.legend(fontsize=9, loc="upper right")
    
    # Panel 3: Reconstructed fraction map
    im2 = axes[1, 0].imshow(f1_recon, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[1, 0].set_title(
        f"Reconstructed: Species-1 Fraction\nPSNR={psnr_val:.1f} dB, SSIM={ssim_val:.4f}",
        fontsize=12,
    )
    plt.colorbar(im2, ax=axes[1, 0], label="$f_1$")
    
    # Panel 4: Error map
    error = f1_recon - f1_gt
    emax = max(abs(error.min()), abs(error.max()), 0.05)
    im3 = axes[1, 1].imshow(error, cmap="RdBu_r", vmin=-emax, vmax=emax, origin="lower")
    axes[1, 1].set_title(f"Error (Recon − GT), MAE={mae_f1:.4f}", fontsize=12)
    plt.colorbar(im3, ax=axes[1, 1], label="$\\Delta f_1$")
    
    fig.suptitle(
        "Task 142: Phasor-based FLIM Lifetime Component Analysis\n"
        f"τ₁={tau1_ns} ns, τ₂={tau2_ns} ns, freq={freq_mhz} MHz, "
        f"{total_photons} photons/px",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = os.path.join(outdir, "reconstruction_result.png")
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {figpath}")
    
    return metrics
