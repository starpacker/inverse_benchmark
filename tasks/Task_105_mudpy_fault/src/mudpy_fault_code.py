"""
mudpy_fault - Finite Fault Inversion using Okada Dislocation Model
===================================================================
From surface displacement data (GPS/InSAR), invert for fault slip distribution
on a discretized fault plane using Okada's analytical solutions.

Physics:
  - Forward: Okada (1985) — surface displacement from rectangular fault patches
  - u(x_obs) = Σ_j G(x_obs, patch_j) × s_j
  - Inverse: Tikhonov-regularized least squares with non-negativity
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.optimize import nnls
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_105_mudpy_fault"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
NX_FAULT    = 10            # fault patches along strike
NY_FAULT    = 5             # fault patches along dip
N_OBS       = 120           # number of GPS/InSAR stations
NOISE_LEVEL = 0.03          # 3% noise on displacements
SEED        = 42
LAMBDA_REG  = 0.0005        # Tikhonov regularization (reduced for well-overdetermined system)

# Fault geometry (simplified planar fault)
FAULT_LENGTH = 100.0        # km along strike
FAULT_WIDTH  = 50.0         # km along dip
FAULT_DEPTH  = 5.0          # km top edge depth
FAULT_DIP    = 15.0         # degrees
FAULT_STRIKE = 0.0          # degrees (N-S)

# Elastic parameters
SHEAR_MOD = 30e9            # Pa — shear modulus
POISSON   = 0.25            # Poisson's ratio

np.random.seed(SEED)


# ====================================================================
# 1. Simplified Okada Green's function
# ====================================================================
def okada_greens_function(obs_x, obs_y, patch_cx, patch_cy, patch_depth,
                          patch_length, patch_width, dip_rad):
    """
    Okada (1985) — simplified closed-form for a point dislocation
    (reverse/thrust fault in elastic half-space).
    
    Uses the Mogi-Okada point-source approximation for dip-slip faults.
    Returns (ux, uy, uz) surface displacement per unit slip.
    """
    dx = obs_x - patch_cx
    dy = obs_y - patch_cy
    d = patch_depth  # depth to center of patch
    
    cos_dip = np.cos(dip_rad)
    sin_dip = np.sin(dip_rad)
    
    # Effective moment area
    A = patch_length * patch_width
    
    # Distance to image source
    R = np.sqrt(dx**2 + dy**2 + d**2)
    R3 = R**3
    R5 = R**5
    
    if R < 0.01:
        return 0.0, 0.0, 0.0
    
    # Poisson ratio factor
    alpha = 1.0 / (2.0 * (1.0 - POISSON))
    
    # Dip-slip point dislocation (Okada approximation)
    # Based on Okada (1992) point source formulas
    # Displacement from a dip-slip dislocation patch
    
    # Horizontal and vertical displacement components
    # For thrust/dip-slip faulting (slip in dip direction)
    factor = A / (4.0 * np.pi)
    
    # Along-dip slip components resolved
    slip_x_comp = -sin_dip * dx / np.sqrt(dx**2 + dy**2 + 1e-10) if np.sqrt(dx**2 + dy**2) > 0.01 else 0.0
    slip_y_comp = -sin_dip * dy / np.sqrt(dx**2 + dy**2 + 1e-10) if np.sqrt(dx**2 + dy**2) > 0.01 else 0.0
    
    # Strain nuclei approach (Mindlin solution)
    ux = factor * (3.0 * dx * d * sin_dip / R5 - 
                   (1.0 - 2.0*POISSON) * dx * sin_dip / (R3))
    uy = factor * (3.0 * dy * d * sin_dip / R5 - 
                   (1.0 - 2.0*POISSON) * dy * sin_dip / (R3))
    uz = factor * (3.0 * d**2 * sin_dip / R5 + 
                   (1.0 - 2.0*POISSON) * sin_dip / R3 +
                   cos_dip * d / R3)
    
    return ux, uy, uz


# ====================================================================
# 2. Build Green's function matrix
# ====================================================================
def build_greens_matrix(obs_coords, patch_params):
    """
    Build the Green's function matrix G such that d = G * s.
    d: displacement vector (3*N_obs,)
    s: slip vector (N_patches,)
    G: Green's function matrix (3*N_obs, N_patches)
    """
    n_obs = obs_coords.shape[0]
    n_patches = len(patch_params)
    G = np.zeros((3 * n_obs, n_patches))

    for j, patch in enumerate(patch_params):
        for i in range(n_obs):
            ux, uy, uz = okada_greens_function(
                obs_coords[i, 0], obs_coords[i, 1],
                patch["cx"], patch["cy"], patch["depth"],
                patch["length"], patch["width"], patch["dip_rad"]
            )
            G[3 * i, j] = ux
            G[3 * i + 1, j] = uy
            G[3 * i + 2, j] = uz

    return G


# ====================================================================
# 3. Ground truth slip distribution
# ====================================================================
def create_gt_slip(nx, ny):
    """
    Create heterogeneous slip distribution on fault plane.
    Simulates an asperity (high-slip zone) surrounded by lower slip.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Main asperity (Gaussian)
    slip = 3.0 * np.exp(-((X - 0.4)**2 / 0.04 + (Y - 0.5)**2 / 0.06))

    # Secondary smaller asperity
    slip += 1.5 * np.exp(-((X - 0.75)**2 / 0.02 + (Y - 0.3)**2 / 0.03))

    # Background low slip
    slip += 0.3 * np.exp(-((X - 0.5)**2 / 0.2 + (Y - 0.5)**2 / 0.2))

    # Ensure non-negative
    slip = np.maximum(slip, 0.0)

    return slip  # shape: (ny, nx), in meters


# ====================================================================
# 4. Setup fault patches
# ====================================================================
def setup_fault_patches(nx, ny, length, width, depth, dip_deg, strike_deg):
    """Discretize fault plane into rectangular patches."""
    dip_rad = np.deg2rad(dip_deg)
    strike_rad = np.deg2rad(strike_deg)

    dx = length / nx
    dy = width / ny

    patches = []
    for j in range(ny):
        for i in range(nx):
            # Along-strike position
            cx = (i + 0.5) * dx - length / 2

            # Along-dip position
            dip_dist = (j + 0.5) * dy
            cy_offset = dip_dist * np.cos(dip_rad)
            cz = depth + dip_dist * np.sin(dip_rad)

            # Rotate by strike
            cx_rot = cx * np.cos(strike_rad)
            cy_rot = cx * np.sin(strike_rad) + cy_offset

            patches.append({
                "cx": cx_rot,
                "cy": cy_rot,
                "depth": cz,
                "length": dx,
                "width": dy,
                "dip_rad": dip_rad,
                "i": i, "j": j,
            })

    return patches


# ====================================================================
# 5. Generate synthetic observations
# ====================================================================
def generate_observations(n_obs, fault_length, fault_width):
    """Generate observation station positions around the fault."""
    # Distribute stations on both sides of the fault
    obs_x = np.random.uniform(-fault_length, fault_length, n_obs)
    obs_y = np.random.uniform(-fault_width * 0.5, fault_width * 2.0, n_obs)
    return np.column_stack([obs_x, obs_y])


# ====================================================================
# 6. Inverse: Tikhonov-regularized NNLS
# ====================================================================
def build_laplacian(nx, ny):
    """Build 2D Laplacian smoothing matrix for fault patches."""
    n = nx * ny
    L = np.zeros((n, n))

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            count = 0

            if i > 0:
                L[idx, idx - 1] = -1.0
                count += 1
            if i < nx - 1:
                L[idx, idx + 1] = -1.0
                count += 1
            if j > 0:
                L[idx, idx - nx] = -1.0
                count += 1
            if j < ny - 1:
                L[idx, idx + nx] = -1.0
                count += 1

            L[idx, idx] = float(count)

    return L


def invert_slip(G, d_obs, nx, ny, lam):
    """
    Tikhonov-regularized non-negative least squares.
    s_hat = argmin ||Gs - d||² + λ||∇s||² subject to s ≥ 0
    """
    L = build_laplacian(nx, ny)

    # Augmented system: [G; sqrt(λ)*L] s = [d; 0]
    G_aug = np.vstack([G, np.sqrt(lam) * L])
    d_aug = np.concatenate([d_obs, np.zeros(nx * ny)])

    # NNLS
    s_hat, residual = nnls(G_aug, d_aug)

    return s_hat


# ====================================================================
# 7. Metrics
# ====================================================================
def compute_metrics(gt, rec):
    """Compute PSNR, SSIM, RMSE for slip distributions using standard definitions."""
    # Standard PSNR: use raw arrays with data_range = gt.max() - gt.min()
    gt_range = gt.max() - gt.min()
    if gt_range < 1e-15:
        gt_range = 1.0
    psnr = float(peak_signal_noise_ratio(gt, rec, data_range=gt_range))

    # SSIM: use raw arrays with proper data_range
    data_range = gt_range
    min_side = min(gt.shape)
    win = min(7, min_side)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    ssim_val = float(ssim(gt, rec, data_range=data_range, win_size=win))

    # CC (correlation coefficient on raw arrays)
    gt_z = gt - gt.mean()
    rec_z = rec - rec.mean()
    denom = np.sqrt(np.sum(gt_z**2) * np.sum(rec_z**2))
    cc = float(np.sum(gt_z * rec_z) / denom) if denom > 1e-15 else 0.0

    # RMSE
    rmse = float(np.sqrt(np.mean((gt - rec)**2)))

    return psnr, ssim_val, cc, rmse


# ====================================================================
# 8. Visualization
# ====================================================================
def plot_results(gt_slip, rec_slip, obs_coords, d_obs, d_pred, metrics, patches):
    """Visualize fault slip and surface displacement."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    nx, ny = NX_FAULT, NY_FAULT

    # 1) GT slip distribution
    ax = axes[0, 0]
    im = ax.imshow(gt_slip, cmap='hot_r', origin='lower', aspect='auto',
                   extent=[0, FAULT_LENGTH, 0, FAULT_WIDTH])
    ax.set_title("Ground Truth Slip Distribution", fontsize=13)
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")
    plt.colorbar(im, ax=ax, label="Slip (m)")

    # 2) Reconstructed slip
    ax = axes[0, 1]
    im = ax.imshow(rec_slip, cmap='hot_r', origin='lower', aspect='auto',
                   extent=[0, FAULT_LENGTH, 0, FAULT_WIDTH])
    ax.set_title(f"Reconstructed Slip\nPSNR={metrics['PSNR']:.1f}dB, "
                 f"SSIM={metrics['SSIM']:.3f}, CC={metrics['CC']:.3f}", fontsize=12)
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")
    plt.colorbar(im, ax=ax, label="Slip (m)")

    # 3) Surface displacement fit — vertical component
    ax = axes[1, 0]
    n_obs = obs_coords.shape[0]
    uz_obs = d_obs[2::3]
    uz_pred = d_pred[2::3]
    sc = ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=uz_obs,
                    cmap='RdBu_r', s=30, edgecolors='k', linewidths=0.3)
    ax.set_title("Observed Vertical Displacement", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    plt.colorbar(sc, ax=ax, label="Uz (m)")

    # 4) Predicted vs observed
    ax = axes[1, 1]
    sc = ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=uz_pred,
                    cmap='RdBu_r', s=30, edgecolors='k', linewidths=0.3,
                    vmin=uz_obs.min(), vmax=uz_obs.max())
    ax.set_title("Predicted Vertical Displacement", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    plt.colorbar(sc, ax=ax, label="Uz (m)")

    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ====================================================================
# 9. Main
# ====================================================================
def main():
    print("=" * 60)
    print("Task 105: Finite Fault Inversion (Okada Model)")
    print("=" * 60)

    # 1) Setup fault patches
    print("[1] Setting up fault plane ...")
    patches = setup_fault_patches(NX_FAULT, NY_FAULT, FAULT_LENGTH, FAULT_WIDTH,
                                  FAULT_DEPTH, FAULT_DIP, FAULT_STRIKE)
    n_patches = len(patches)
    print(f"    {NX_FAULT}×{NY_FAULT} = {n_patches} patches")
    print(f"    Fault: {FAULT_LENGTH}×{FAULT_WIDTH} km, dip={FAULT_DIP}°")

    # 2) Ground truth slip
    print("[2] Creating ground truth slip distribution ...")
    gt_slip = create_gt_slip(NX_FAULT, NY_FAULT)
    gt_slip_vec = gt_slip.ravel()
    print(f"    Max slip: {gt_slip.max():.2f} m")
    print(f"    Mean slip: {gt_slip.mean():.2f} m")

    # 3) Observation stations
    print("[3] Generating observation stations ...")
    obs_coords = generate_observations(N_OBS, FAULT_LENGTH, FAULT_WIDTH)
    print(f"    {N_OBS} stations")

    # 4) Build Green's matrix
    print("[4] Building Green's function matrix ...")
    t0 = time.time()
    G = build_greens_matrix(obs_coords, patches)
    t_green = time.time() - t0
    print(f"    G shape: {G.shape}, built in {t_green:.1f}s")

    # 5) Forward: synthetic data
    print("[5] Computing synthetic displacements ...")
    d_true = G @ gt_slip_vec
    noise = NOISE_LEVEL * np.abs(d_true).max() * np.random.randn(len(d_true))
    d_obs = d_true + noise
    print(f"    Max displacement: {np.abs(d_true).max():.4f} m")

    # 6) Inverse
    print(f"[6] Inverting for slip (λ={LAMBDA_REG}) ...")
    t0 = time.time()
    s_hat = invert_slip(G, d_obs, NX_FAULT, NY_FAULT, LAMBDA_REG)
    t_inv = time.time() - t0
    rec_slip = s_hat.reshape(NY_FAULT, NX_FAULT)
    print(f"    Inversion: {t_inv:.1f}s")
    print(f"    Reconstructed max slip: {rec_slip.max():.2f} m")

    # 7) Predicted data
    d_pred = G @ s_hat

    # 8) Metrics
    print("[7] Computing metrics ...")
    psnr, ssim_val, cc, rmse = compute_metrics(gt_slip, rec_slip)
    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    SSIM = {ssim_val:.4f}")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.4f} m")

    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "CC": float(cc),
        "RMSE": float(rmse),
    }

    # 9) Save
    print("[8] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_slip)
        np.save(os.path.join(d, "recon_output.npy"), rec_slip)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # 10) Plot
    print("[9] Plotting ...")
    plot_results(gt_slip, rec_slip, obs_coords, d_obs, d_pred, metrics, patches)

    print(f"\n{'=' * 60}")
    print("Task 105 COMPLETE")
    print(f"{'=' * 60}")
    return metrics


if __name__ == "__main__":
    metrics = main()
