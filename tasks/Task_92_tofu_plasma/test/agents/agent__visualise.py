import matplotlib

matplotlib.use("Agg")

import numpy as np

import matplotlib.pyplot as plt

def _visualise(r_arr, z_arr, gt, sinogram, recon, error, metrics, 
               n_detectors, n_los_per_det, save_path):
    """4-panel figure: GT, sinogram, reconstruction, error map."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent_rz = [r_arr[0], r_arr[-1], z_arr[0], z_arr[-1]]

    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(gt.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title("(a) Ground Truth Emissivity ε(R,Z)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (b) Line-integrated measurements (sinogram-like)
    ax = axes[0, 1]
    n_det = n_detectors
    n_los = n_los_per_det
    sino_2d = sinogram.reshape(n_det, n_los) if len(sinogram) == n_det * n_los \
        else sinogram.reshape(-1, n_los)[:n_det]
    im = ax.imshow(sino_2d, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("(b) Line-Integrated Measurements")
    ax.set_xlabel("LOS index")
    ax.set_ylabel("Detector fan")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (c) Reconstruction
    ax = axes[1, 0]
    im = ax.imshow(recon.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title(f"(c) Reconstruction  PSNR={metrics['PSNR']:.1f} dB  "
                 f"SSIM={metrics['SSIM']:.3f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (d) Error map
    ax = axes[1, 1]
    im = ax.imshow(error.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="seismic",
                   vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    ax.set_title(f"(d) Error Map  RMSE={metrics['RMSE']:.4f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Plasma/Fusion Tomography — Tokamak Emissivity Reconstruction",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualisation → {save_path}")
