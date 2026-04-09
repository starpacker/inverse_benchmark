"""
simpeg — Gravity Anomaly Inversion
====================================
Task: Recover 3D subsurface density contrast model from surface
      gravity anomaly measurements.

Inverse Problem:
    Given surface gravity anomaly data g_z(x,y), recover 3D
    density contrast distribution Δρ(x,y,z) in the subsurface.

Forward Model (SimPEG):
    g_z = G · Δρ   where G is the gravity kernel (Green's function
    for a gravitational point mass) computed by SimPEG's potential
    field simulation engine.

Inverse Solver:
    Tikhonov-regularised Gauss-Newton inversion using SimPEG's
    inverse problem framework with depth weighting and smoothness
    regularisation.

Repo: https://github.com/simpeg/simpeg
Paper: Cockett et al. (2015), Computers & Geosciences, 85, 142–154.
       doi:10.1016/j.cageo.2015.09.015

Usage:
    /data/yjh/geo_env/bin/python simpeg_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ── SimPEG library imports ──────────────────────────────────
from simpeg import (
    maps, data_misfit, regularization, optimization,
    inverse_problem, inversion, directives, data,
)
from simpeg.potential_fields import gravity
from discretize import TensorMesh

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mesh parameters (3D tensor mesh)
N_CELLS_X = 20
N_CELLS_Y = 20
N_CELLS_Z = 10
CELL_SIZE_X = 50.0   # m
CELL_SIZE_Y = 50.0   # m
CELL_SIZE_Z = 25.0   # m

# Survey parameters
N_RX_X = 15
N_RX_Y = 15
RX_HEIGHT = 1.0     # m above surface
NOISE_FLOOR = 0.01  # mGal absolute noise
NOISE_PCT = 0.02    # 2% relative noise

# Ground truth: density anomaly
GT_DENSITY = 0.5     # g/cm³ contrast
GT_CENTER = [500, 500, -150]  # m (x, y, z)
GT_RADIUS = 100.0    # m

SEED = 42


# ═══════════════════════════════════════════════════════════
# 2. Mesh & Survey Setup (SimPEG objects)
# ═══════════════════════════════════════════════════════════
def create_mesh():
    """Create 3D tensor mesh using discretize."""
    hx = np.ones(N_CELLS_X) * CELL_SIZE_X
    hy = np.ones(N_CELLS_Y) * CELL_SIZE_Y
    hz = np.ones(N_CELLS_Z) * CELL_SIZE_Z
    mesh = TensorMesh([hx, hy, hz], origin='CC0')
    # Shift so surface is at z=0
    mesh.origin = np.array([
        -N_CELLS_X * CELL_SIZE_X / 2,
        -N_CELLS_Y * CELL_SIZE_Y / 2,
        -N_CELLS_Z * CELL_SIZE_Z,
    ])
    return mesh


def create_survey():
    """Create surface gravity survey using SimPEG gravity module."""
    rx_x = np.linspace(
        -N_CELLS_X * CELL_SIZE_X / 2 * 0.7,
        N_CELLS_X * CELL_SIZE_X / 2 * 0.7,
        N_RX_X
    )
    rx_y = np.linspace(
        -N_CELLS_Y * CELL_SIZE_Y / 2 * 0.7,
        N_CELLS_Y * CELL_SIZE_Y / 2 * 0.7,
        N_RX_Y
    )
    rx_xx, rx_yy = np.meshgrid(rx_x, rx_y)
    rx_locs = np.c_[
        rx_xx.ravel(),
        rx_yy.ravel(),
        np.full(N_RX_X * N_RX_Y, RX_HEIGHT)
    ]

    # Create receiver list
    receiver_list = [gravity.receivers.Point(rx_locs, components=["gz"])]

    # Create source (gravity is a potential field — no "source" per se)
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)

    # Create survey
    survey = gravity.survey.Survey(source_field)

    return survey, rx_locs


# ═══════════════════════════════════════════════════════════
# 3. Ground Truth & Forward Operator
# ═══════════════════════════════════════════════════════════
def create_ground_truth(mesh):
    """
    Create ground truth density model: spherical anomaly.
    """
    cc = mesh.cell_centers
    dist = np.sqrt(
        (cc[:, 0] - GT_CENTER[0]) ** 2 +
        (cc[:, 1] - GT_CENTER[1]) ** 2 +
        (cc[:, 2] - GT_CENTER[2]) ** 2
    )
    model_gt = np.zeros(mesh.n_cells)
    model_gt[dist < GT_RADIUS] = GT_DENSITY

    # Add a smaller secondary anomaly
    dist2 = np.sqrt(
        (cc[:, 0] - (GT_CENTER[0] + 200)) ** 2 +
        (cc[:, 1] - (GT_CENTER[1] - 150)) ** 2 +
        (cc[:, 2] - (GT_CENTER[2] - 50)) ** 2
    )
    model_gt[dist2 < GT_RADIUS * 0.6] = -0.3

    return model_gt


def forward_operator(mesh, survey, model):
    """
    SimPEG gravity forward simulation.

    Uses the integral equation approach: each mesh cell is treated
    as a uniform-density prism, and the vertical gravity component
    gz is computed analytically using the prism formula of Blakely (1996).

    Parameters
    ----------
    mesh : TensorMesh   3D discretize mesh.
    survey : Survey      SimPEG gravity survey.
    model : np.ndarray   Density contrast vector (g/cm³).

    Returns
    -------
    d_pred : np.ndarray  Predicted gravity anomaly [mGal].
    """
    # Identity map (model is density directly)
    model_map = maps.IdentityMap(nP=mesh.n_cells)

    simulation = gravity.simulation.Simulation3DIntegral(
        mesh=mesh,
        survey=survey,
        rhoMap=model_map,
        ind_active=np.ones(mesh.n_cells, dtype=bool),
        store_sensitivities="ram",
    )

    d_pred = simulation.dpred(model)
    return d_pred, simulation


# ═══════════════════════════════════════════════════════════
# 4. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic gravity survey data with SimPEG."""
    print("[DATA] Creating 3D tensor mesh ...")
    mesh = create_mesh()
    print(f"[DATA] Mesh: {mesh.shape_cells} cells, "
          f"extent {mesh.nodes_x.ptp():.0f}×{mesh.nodes_y.ptp():.0f}×"
          f"{mesh.nodes_z.ptp():.0f} m")

    print("[DATA] Creating gravity survey ...")
    survey, rx_locs = create_survey()
    print(f"[DATA] {rx_locs.shape[0]} receivers at z={RX_HEIGHT} m")

    print("[DATA] Building ground truth density model ...")
    model_gt = create_ground_truth(mesh)
    print(f"[DATA] Anomaly: {(model_gt != 0).sum()} active cells, "
          f"Δρ range [{model_gt.min():.2f}, {model_gt.max():.2f}] g/cm³")

    print("[DATA] Running SimPEG forward simulation ...")
    d_clean, simulation = forward_operator(mesh, survey, model_gt)
    print(f"[DATA] g_z range: [{d_clean.min():.4f}, {d_clean.max():.4f}] mGal")

    # Add noise
    rng = np.random.default_rng(SEED)
    std = NOISE_FLOOR + NOISE_PCT * np.abs(d_clean)
    noise = std * rng.standard_normal(len(d_clean))
    d_noisy = d_clean + noise

    return mesh, survey, model_gt, d_clean, d_noisy, std, rx_locs, simulation


# ═══════════════════════════════════════════════════════════
# 5. Inverse Solver (SimPEG Inversion)
# ═══════════════════════════════════════════════════════════
def reconstruct(mesh, survey, d_noisy, std, simulation):
    """
    SimPEG gravity inversion.

    Uses:
      - L2 data misfit
      - Tikhonov regularisation (smoothness + smallness)
      - Depth weighting
      - IRLS for compact recovery

    Parameters
    ----------
    mesh : TensorMesh
    survey : Survey
    d_noisy : np.ndarray   Noisy gravity data.
    std : np.ndarray       Data uncertainty.
    simulation : Simulation

    Returns
    -------
    model_rec : np.ndarray  Recovered density model.
    """
    # Data object
    data_obj = data.Data(survey, dobs=d_noisy, standard_deviation=std)

    # Data misfit
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)

    # Model map
    model_map = maps.IdentityMap(nP=mesh.n_cells)

    # Regularisation
    reg = regularization.WeightedLeastSquares(
        mesh,
        alpha_s=1e-4,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
    )

    # Optimisation
    opt = optimization.InexactGaussNewton(
        maxIter=10, maxIterLS=8, maxIterCG=20, tolCG=1e-3
    )
    opt.remember("xc")

    # Inverse problem
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    # Directives
    target = directives.TargetMisfit()
    beta_est = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    save_dict = directives.SaveOutputDictEveryIteration(save_txt=False)
    update_jacobi = directives.UpdatePreconditioner()

    directives_list = [
        target, beta_est, beta_schedule, save_dict, update_jacobi
    ]

    # Run inversion
    print("[RECON] Running SimPEG Gauss-Newton inversion ...")
    inv = inversion.BaseInversion(inv_prob, directives_list)

    m0 = np.zeros(mesh.n_cells)  # starting model
    model_rec = inv.run(m0)

    return model_rec


# ═══════════════════════════════════════════════════════════
# 6. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(model_gt, model_rec, d_clean, d_pred_rec, mesh):
    """Compute inversion quality metrics."""
    from skimage.metrics import structural_similarity as ssim_fn

    # Model-space metrics (3D → reshape to evaluate per-layer)
    nx, ny, nz = mesh.shape_cells
    gt_3d = model_gt.reshape((nx, ny, nz), order='F')
    rec_3d = model_rec.reshape((nx, ny, nz), order='F')

    # Take a horizontal slice at anomaly depth
    iz_anom = nz // 2
    gt_slice = gt_3d[:, :, iz_anom]
    rec_slice = rec_3d[:, :, iz_anom]

    data_range = gt_slice.max() - gt_slice.min()
    if data_range < 1e-12:
        data_range = 1.0

    mse = np.mean((gt_slice - rec_slice) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_slice, rec_slice, data_range=data_range))
    cc_slice = float(np.corrcoef(gt_slice.ravel(), rec_slice.ravel())[0, 1])

    # Volume metrics
    cc_vol = float(np.corrcoef(model_gt, model_rec)[0, 1])
    re_vol = float(np.linalg.norm(model_gt - model_rec) /
                   max(np.linalg.norm(model_gt), 1e-12))

    # Data fit metrics
    residual = d_clean - d_pred_rec
    rmse_data = float(np.sqrt(np.mean(residual ** 2)))
    cc_data = float(np.corrcoef(d_clean, d_pred_rec)[0, 1])

    metrics = {
        "PSNR_slice": psnr,
        "SSIM_slice": ssim_val,
        "CC_slice": cc_slice,
        "CC_volume": cc_vol,
        "RE_volume": re_vol,
        "RMSE_data_mGal": rmse_data,
        "CC_data": cc_data,
    }
    return metrics


# ═══════════════════════════════════════════════════════════
# 7. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(mesh, model_gt, model_rec, rx_locs,
                      d_clean, d_noisy, d_rec, metrics, save_path):
    nx, ny, nz = mesh.shape_cells
    gt_3d = model_gt.reshape((nx, ny, nz), order='F')
    rec_3d = model_rec.reshape((nx, ny, nz), order='F')

    iz = nz // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmax = max(np.abs(gt_3d).max(), 0.1)

    # (a) GT slice
    im = axes[0, 0].imshow(gt_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'(a) GT Density (z-slice {iz})')
    plt.colorbar(im, ax=axes[0, 0], label='Δρ [g/cm³]')

    # (b) Reconstructed slice
    im = axes[0, 1].imshow(rec_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 1].set_title('(b) SimPEG Inversion')
    plt.colorbar(im, ax=axes[0, 1], label='Δρ [g/cm³]')

    # (c) Error
    err = gt_3d[:, :, iz] - rec_3d[:, :, iz]
    im = axes[0, 2].imshow(err.T, cmap='RdBu_r',
                            vmin=-vmax/2, vmax=vmax/2, origin='lower')
    axes[0, 2].set_title('(c) Error')
    plt.colorbar(im, ax=axes[0, 2], label='Δρ error')

    # (d) Observed data map
    n_rx = int(np.sqrt(len(d_clean)))
    if n_rx ** 2 == len(d_clean):
        d_map = d_clean.reshape(n_rx, n_rx)
        axes[1, 0].imshow(d_map, cmap='viridis', origin='lower')
    else:
        axes[1, 0].scatter(rx_locs[:, 0], rx_locs[:, 1],
                           c=d_clean, cmap='viridis', s=20)
    axes[1, 0].set_title('(d) Gravity Anomaly (GT)')

    # (e) Data fit
    axes[1, 1].plot(d_clean, d_rec, 'b.', ms=3)
    lims = [min(d_clean.min(), d_rec.min()),
            max(d_clean.max(), d_rec.max())]
    axes[1, 1].plot(lims, lims, 'k--', lw=0.5)
    axes[1, 1].set_xlabel('True g_z [mGal]')
    axes[1, 1].set_ylabel('Predicted g_z [mGal]')
    axes[1, 1].set_title(f'(e) Data Fit  CC={metrics["CC_data"]:.4f}')

    # (f) Depth profile
    axes[1, 2].plot(gt_3d[nx//2, ny//2, :], range(nz), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_3d[nx//2, ny//2, :], range(nz), 'r--', lw=2, label='Inv')
    axes[1, 2].set_xlabel('Δρ [g/cm³]')
    axes[1, 2].set_ylabel('Depth index')
    axes[1, 2].set_title('(f) Depth Profile')
    axes[1, 2].legend()
    axes[1, 2].invert_yaxis()

    fig.suptitle(
        f"SimPEG — Gravity Anomaly Inversion\n"
        f"PSNR={metrics['PSNR_slice']:.1f} dB  |  "
        f"SSIM={metrics['SSIM_slice']:.4f}  |  "
        f"CC_vol={metrics['CC_volume']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 8. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  SimPEG — Gravity Anomaly Inversion")
    print("=" * 65)

    mesh, survey, model_gt, d_clean, d_noisy, std, rx_locs, sim = \
        load_or_generate_data()

    print("\n[RECON] Running SimPEG Gauss-Newton inversion ...")
    model_rec = reconstruct(mesh, survey, d_noisy, std, sim)

    # Predicted data from recovered model
    d_rec = sim.dpred(model_rec)

    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(model_gt, model_rec, d_clean, d_rec, mesh)
    for k, v in sorted(metrics.items()):
        print(f"  {k:25s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), model_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), model_gt)

    visualize_results(mesh, model_gt, model_rec, rx_locs,
                      d_clean, d_noisy, d_rec, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 65)
    print("  DONE — SimPEG gravity inversion benchmark complete")
    print("=" * 65)
