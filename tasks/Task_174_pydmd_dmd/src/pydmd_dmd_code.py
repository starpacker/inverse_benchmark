"""
Task 174: pydmd_dmd — Dynamic Mode Decomposition (DMD)

Inverse problem: Recover spatial modes and temporal dynamics from
high-dimensional spatiotemporal flow-field data (system identification).

Forward model:
    X(x, t) = Σ_i φ_i(x) · exp((σ_i + jω_i) · t) · b_i
where φ_i are spatial modes, σ_i decay rates, ω_i frequencies, b_i amplitudes.

Inverse solver:
    Standard DMD with truncated SVD decomposes snapshot matrix X ≈ Φ Λ^k B
    recovering spatial modes Φ and discrete eigenvalues μ = exp((σ + jω)·dt).

Data generation uses 2 oscillatory modes (4 spatial patterns for the real
and imaginary parts of each conjugate pair), producing a rank-4 snapshot
matrix.  DMD with svd_rank=4 recovers the 4 discrete eigenvalues as 2
conjugate pairs.
"""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydmd import DMD

# ---------------------------------------------------------------------------
# Paths — all relative to THIS script
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical / numerical parameters
# ---------------------------------------------------------------------------
NX, NY = 64, 64          # spatial grid
NT = 200                  # number of time snapshots
DT = 0.02                # time step  (s)
SNR_DB = 30               # signal-to-noise ratio for additive Gaussian noise
SEED = 42
SVD_RANK = 4              # 2 conjugate pairs → 4 DMD eigenvalues


def synthesize_data():
    """Create a rank-4 spatiotemporal field from 2 oscillatory modes.

    Each oscillatory mode (complex eigenvalue λ = σ + jω) produces a
    conjugate pair of discrete eigenvalues μ, μ*.  Taking the real part of
    the dynamics couples two spatial patterns (cos/sin components), making
    the snapshot matrix rank 4 for 2 such modes.

    Returns
    -------
    gt_field   : ndarray (NX*NY, NT) — ground-truth snapshot matrix
    noisy_field: ndarray (NX*NY, NT) — noisy observations
    meta       : dict with spatial coords, time, true eigenvalues, etc.
    """
    rng = np.random.default_rng(SEED)

    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NY)
    k = np.arange(NT)
    t = k * DT
    Xg, Yg = np.meshgrid(x, y, indexing="ij")  # (NX, NY)

    # --- 4 spatial patterns (2 pairs: cos/sin parts) ---
    phi1 = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)       # mode-1 real
    phi2 = np.sin(2 * np.pi * Xg) * np.cos(np.pi * Yg)   # mode-1 imag
    phi3 = np.cos(np.pi * Xg) * np.sin(2 * np.pi * Yg)   # mode-2 real
    phi4 = np.cos(2 * np.pi * Xg) * np.cos(2 * np.pi * Yg)  # mode-2 imag

    # --- Continuous-time eigenvalues ---
    omega1, sigma1 = 2.3, -0.1     # frequency, decay rate (mode 1)
    omega2, sigma2 = 5.7, -0.05    # frequency, decay rate (mode 2)
    lam1 = sigma1 + 1j * omega1
    lam2 = sigma2 + 1j * omega2

    # Discrete-time eigenvalues
    mu1 = np.exp(lam1 * DT)
    mu2 = np.exp(lam2 * DT)

    # Temporal dynamics (real & imaginary parts from each conjugate pair)
    c1_re = np.real(mu1 ** k)   # exp(σ₁Δt·k) · cos(ω₁Δt·k)
    c1_im = np.imag(mu1 ** k)   # exp(σ₁Δt·k) · sin(ω₁Δt·k)
    c2_re = np.real(mu2 ** k)
    c2_im = np.imag(mu2 ** k)

    # Snapshot matrix: X = Σ φ_i ⊗ c_i  (rank 4)
    gt_field = (np.outer(phi1.ravel(), c1_re)
                + np.outer(phi2.ravel(), c1_im)
                + np.outer(phi3.ravel(), c2_re)
                + np.outer(phi4.ravel(), c2_im))

    # Additive Gaussian noise
    signal_power = np.mean(gt_field ** 2)
    noise_power = signal_power / (10 ** (SNR_DB / 10))
    noise = rng.normal(0, np.sqrt(noise_power), gt_field.shape)
    noisy_field = gt_field + noise

    meta = dict(
        x=x, y=y, t=t, dt=DT,
        true_continuous_eigenvalues=np.array([lam1, lam2]),
        true_discrete_eigenvalues=np.array([mu1, mu2]),
        omega=[omega1, omega2],
        sigma=[sigma1, sigma2],
    )
    return gt_field, noisy_field, meta


def run_dmd(noisy_field):
    """Fit standard DMD and return reconstruction + model."""
    dmd = DMD(svd_rank=SVD_RANK)
    dmd.fit(noisy_field)
    recon = dmd.reconstructed_data.real
    return dmd, recon


def compute_metrics(gt, recon):
    """PSNR, correlation coefficient, MSE between GT and reconstruction."""
    gt64 = gt.astype(np.float64)
    re64 = recon.astype(np.float64)

    # Align time axis (reconstructed_data may differ by 1 column)
    min_t = min(gt64.shape[1], re64.shape[1])
    gt64 = gt64[:, :min_t]
    re64 = re64[:, :min_t]

    mse = np.mean((gt64 - re64) ** 2)
    data_range = gt64.max() - gt64.min()
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float("inf")

    cc = float(np.corrcoef(gt64.ravel(), re64.ravel())[0, 1])

    return {
        "psnr_db": round(float(psnr), 4),
        "correlation_coefficient": round(cc, 6),
        "mse": float(f"{mse:.10e}"),
    }


def compute_eigenvalue_error(dmd, true_discrete_eigs):
    """Relative error of recovered discrete eigenvalues vs ground truth.

    Each GT eigenvalue (and its conjugate) is matched to the nearest
    recovered eigenvalue.
    """
    recovered = dmd.eigs
    gt_all = np.concatenate([true_discrete_eigs, true_discrete_eigs.conj()])
    errors = []
    for gt_e in gt_all:
        dists = np.abs(recovered - gt_e)
        idx = np.argmin(dists)
        rel_err = float(np.abs(recovered[idx] - gt_e) / np.abs(gt_e))
        errors.append(round(rel_err, 8))
    return errors


def visualize(gt_field, noisy_field, recon, save_path):
    """4-panel figure at t = 0."""
    gt_img = gt_field[:, 0].reshape(NX, NY)
    noisy_img = noisy_field[:, 0].reshape(NX, NY)
    recon_img = recon[:, 0].reshape(NX, NY)
    err_img = np.abs(gt_img - recon_img)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    titles = [
        "(a) Ground Truth (t=0)",
        "(b) Noisy Input (t=0)",
        "(c) DMD Reconstruction (t=0)",
        "(d) Error |GT − Recon|",
    ]
    images = [gt_img, noisy_img, recon_img, err_img]
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "hot"]

    vmin = min(gt_img.min(), noisy_img.min(), recon_img.min())
    vmax = max(gt_img.max(), noisy_img.max(), recon_img.max())

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        if cmap == "hot":
            im = ax.imshow(img.T, origin="lower", cmap=cmap, aspect="equal")
        else:
            im = ax.imshow(
                img.T, origin="lower", cmap=cmap, aspect="equal",
                vmin=vmin, vmax=vmax,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Task 174 — Dynamic Mode Decomposition (PyDMD)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved visualization → {save_path}")


# ===================================================================
def main():
    # 1. Synthesize data
    gt_field, noisy_field, meta = synthesize_data()
    print(f"[INFO] Snapshot matrix: {gt_field.shape}  "
          f"(spatial={NX*NY}, time={NT})")
    print(f"[INFO] GT range: [{gt_field.min():.4f}, {gt_field.max():.4f}]")

    # 2. DMD inverse
    dmd, recon = run_dmd(noisy_field)
    print(f"[INFO] DMD modes shape : {dmd.modes.shape}")
    print(f"[INFO] DMD eigenvalues : {dmd.eigs}")
    print(f"[INFO] Reconstruction  : {recon.shape}")

    # 3. Evaluate
    metrics = compute_metrics(gt_field, recon)
    eig_errors = compute_eigenvalue_error(
        dmd, meta["true_discrete_eigenvalues"]
    )
    metrics["eigenvalue_relative_errors"] = eig_errors
    metrics["n_modes"] = int(len(dmd.eigs))
    metrics["svd_rank"] = SVD_RANK

    print(f"\n[METRICS] PSNR  = {metrics['psnr_db']:.2f} dB")
    print(f"[METRICS] CC    = {metrics['correlation_coefficient']:.6f}")
    print(f"[METRICS] MSE   = {metrics['mse']:.2e}")
    print(f"[METRICS] Eig. rel. err = {eig_errors}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[INFO] Saved metrics → {metrics_path}")

    # 4. Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_field)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon)
    print("[INFO] Saved ground_truth.npy and reconstruction.npy")

    # 5. Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize(gt_field, noisy_field, recon, vis_path)

    print("\n[DONE] Task 174 pydmd_dmd completed successfully.")


if __name__ == "__main__":
    main()
