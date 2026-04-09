"""
myptv — 3D Particle Tracking Velocimetry (PTV)
===============================================
Task: Reconstruct 3D particle positions from multi-camera
      2D particle image projections via stereo triangulation
Repo: https://github.com/ronshnapp/MyPTV

Inverse Problem:
    Forward: Given 3D particle positions X_j, project onto each camera i
             via pinhole model:  x_ij = K_i @ [R_i | t_i] @ X_j
             (3D→2D projection, then add detection noise)
    Inverse: Given 2D detections {x_ij} across cameras, triangulate
             3D positions using Direct Linear Transform (DLT) /
             least-squares ray intersection

Usage:
    /data/yjh/myptv_env/bin/python myptv_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# Measurement volume [mm]
VOL_MIN = np.array([-50.0, -50.0, -50.0])
VOL_MAX = np.array([50.0, 50.0, 50.0])

# Particle parameters
N_PARTICLES = 200       # Number of tracer particles
NOISE_STD_PX = 0.5      # 2D detection noise [pixels]

# Camera parameters
N_CAMERAS = 4
IMAGE_W = 1024           # Sensor width [pixels]
IMAGE_H = 1024           # Sensor height [pixels]
FOCAL_LENGTH_PX = 2000.0 # Focal length [pixels]


# ═══════════════════════════════════════════════════════════
# 2. Camera Model & Calibration
# ═══════════════════════════════════════════════════════════
def rotation_matrix(axis, angle_deg):
    """Rodrigues rotation about a given axis ('x','y','z')."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"Unknown axis: {axis}")


def setup_cameras():
    """
    Set up N_CAMERAS cameras arranged around the measurement volume.

    Each camera is defined by:
        K: 3×3 intrinsic matrix
        R: 3×3 rotation (world→camera)
        t: 3×1 translation (world→camera)
        P: 3×4 projection matrix  P = K @ [R | t]

    Returns list of dicts with keys: K, R, t, P, cam_pos
    """
    cameras = []

    # Camera positions: placed at ~300 mm from the volume center,
    # looking inward from different azimuth angles
    cam_distance = 300.0
    azimuth_angles = [30.0, 120.0, 210.0, 300.0]  # degrees
    elevation_angle = 20.0  # slight downward tilt

    for i in range(N_CAMERAS):
        az = np.radians(azimuth_angles[i])
        el = np.radians(elevation_angle)

        # Camera position in world coordinates
        cam_pos = cam_distance * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])

        # Camera looks at the origin (volume center)
        look_at = np.array([0.0, 0.0, 0.0])
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        # World up
        up = np.array([0.0, 0.0, 1.0])

        # Camera coordinate axes
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # Rotation matrix (world→camera): rows are camera axes
        R = np.array([right, cam_up, forward])  # 3×3

        # Translation: t = -R @ cam_pos
        t = -R @ cam_pos  # 3×1

        # Intrinsic matrix (pinhole)
        K = np.array([
            [FOCAL_LENGTH_PX, 0.0, IMAGE_W / 2.0],
            [0.0, FOCAL_LENGTH_PX, IMAGE_H / 2.0],
            [0.0, 0.0, 1.0]
        ])

        # Projection matrix P = K @ [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])  # 3×4
        P = K @ Rt  # 3×4

        cameras.append({
            'K': K, 'R': R, 't': t, 'P': P,
            'cam_pos': cam_pos, 'id': i
        })

    return cameras


# ═══════════════════════════════════════════════════════════
# 3. Ground Truth Generation
# ═══════════════════════════════════════════════════════════
def generate_particles():
    """
    Generate N_PARTICLES random 3D positions uniformly
    distributed in the measurement volume.

    Returns: (N_PARTICLES, 3) array of 3D positions [mm]
    """
    positions = np.random.uniform(
        VOL_MIN, VOL_MAX, size=(N_PARTICLES, 3)
    )
    return positions


# ═══════════════════════════════════════════════════════════
# 4. Forward Operator: 3D → 2D Projection
# ═══════════════════════════════════════════════════════════
def project_points(particles_3d, camera):
    """
    Project 3D points onto a camera's image plane.

    Args:
        particles_3d: (N, 3) array of 3D positions
        camera: dict with projection matrix P

    Returns:
        uv: (N, 2) array of 2D pixel coordinates
        visible: (N,) boolean mask — True if point is in front of camera
                 and within image bounds
    """
    P = camera['P']
    N = particles_3d.shape[0]

    # Homogeneous coordinates
    X_hom = np.hstack([particles_3d, np.ones((N, 1))])  # (N, 4)

    # Project: s * [u, v, 1]^T = P @ [X, Y, Z, 1]^T
    proj = (P @ X_hom.T).T  # (N, 3)

    # Perspective divide
    depth = proj[:, 2]
    uv = proj[:, :2] / depth[:, np.newaxis]

    # Visibility check: positive depth + within image bounds
    visible = (
        (depth > 0) &
        (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_H)
    )

    return uv, visible


def forward_operator(particles_3d, cameras, noise_std=NOISE_STD_PX):
    """
    Full forward model: project all particles onto all cameras
    and add Gaussian detection noise.

    Returns:
        detections: list of N_CAMERAS arrays, each (M_i, 2) pixel coords
        visibility: list of N_CAMERAS boolean masks, each (N_PARTICLES,)
        clean_projections: list of noise-free projections (for reference)
    """
    detections = []
    visibility = []
    clean_projections = []

    for cam in cameras:
        uv_clean, vis = project_points(particles_3d, cam)

        # Add Gaussian noise to detections
        noise = np.random.normal(0, noise_std, uv_clean.shape)
        uv_noisy = uv_clean + noise

        # Re-check bounds after noise
        vis_noisy = (
            vis &
            (uv_noisy[:, 0] >= 0) & (uv_noisy[:, 0] < IMAGE_W) &
            (uv_noisy[:, 1] >= 0) & (uv_noisy[:, 1] < IMAGE_H)
        )

        detections.append(uv_noisy)
        visibility.append(vis_noisy)
        clean_projections.append(uv_clean)

    return detections, visibility, clean_projections


# ═══════════════════════════════════════════════════════════
# 5. Inverse Solver: 2D Detections → 3D Triangulation
# ═══════════════════════════════════════════════════════════
def triangulate_dlt(uv_list, P_list):
    """
    Triangulate a single 3D point from its 2D projections
    in multiple cameras using Direct Linear Transform (DLT).

    For each camera i with projection matrix P_i and detection (u_i, v_i):
        u_i * P_i[2,:] - P_i[0,:] = 0
        v_i * P_i[2,:] - P_i[1,:] = 0

    Stack into A @ X = 0, solve via SVD.

    Args:
        uv_list: list of (2,) arrays — 2D coords in each camera
        P_list: list of (3,4) projection matrices

    Returns:
        X_3d: (3,) reconstructed 3D position
    """
    n_views = len(uv_list)
    A = np.zeros((2 * n_views, 4))

    for i, (uv, P) in enumerate(zip(uv_list, P_list)):
        u, v = uv
        A[2 * i]     = u * P[2, :] - P[0, :]
        A[2 * i + 1] = v * P[2, :] - P[1, :]

    # Solve A @ X = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1, :]  # last row of V^T

    # Convert from homogeneous
    X_3d = X_hom[:3] / X_hom[3]
    return X_3d


def compute_reprojection_error(X_3d, uv_list, P_list):
    """Compute mean reprojection error for a triangulated point."""
    X_hom = np.append(X_3d, 1.0)
    errors = []
    for uv, P in zip(uv_list, P_list):
        proj = P @ X_hom
        proj_2d = proj[:2] / proj[2]
        err = np.linalg.norm(proj_2d - uv)
        errors.append(err)
    return np.mean(errors)


def inverse_operator(detections, visibility, cameras):
    """
    Reconstruct 3D particle positions from multi-camera 2D detections.

    For each particle visible in ≥2 cameras, gather its detections
    and triangulate via DLT.

    Args:
        detections: list of (N_PARTICLES, 2) arrays per camera
        visibility: list of (N_PARTICLES,) boolean masks per camera
        cameras: list of camera dicts

    Returns:
        recon_3d: (N_PARTICLES, 3) array (NaN for unrecoverable particles)
        n_views_per_particle: (N_PARTICLES,) number of views used
        reproj_errors: (N_PARTICLES,) reprojection errors
    """
    N = detections[0].shape[0]
    recon_3d = np.full((N, 3), np.nan)
    n_views = np.zeros(N, dtype=int)
    reproj_errors = np.full(N, np.nan)

    for j in range(N):
        # Gather detections for particle j across all cameras
        uv_list = []
        P_list = []

        for i, cam in enumerate(cameras):
            if visibility[i][j]:
                uv_list.append(detections[i][j])
                P_list.append(cam['P'])

        n_views[j] = len(uv_list)

        if len(uv_list) >= 2:
            X_3d = triangulate_dlt(uv_list, P_list)
            recon_3d[j] = X_3d
            reproj_errors[j] = compute_reprojection_error(
                X_3d, uv_list, P_list
            )

    return recon_3d, n_views, reproj_errors


# ═══════════════════════════════════════════════════════════
# 6. Metrics Computation
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_3d, recon_3d, n_views):
    """
    Compute reconstruction quality metrics.

    Args:
        gt_3d: (N, 3) ground truth positions
        recon_3d: (N, 3) reconstructed positions (may contain NaN)
        n_views: (N,) number of cameras that saw each particle

    Returns:
        dict of metrics
    """
    # Mask for successfully reconstructed particles (≥2 views)
    valid = ~np.isnan(recon_3d[:, 0])
    n_valid = np.sum(valid)
    n_total = len(gt_3d)

    if n_valid == 0:
        return {
            "rmse_3d_mm": float('inf'),
            "mean_error_mm": float('inf'),
            "median_error_mm": float('inf'),
            "max_error_mm": float('inf'),
            "correlation_x": 0.0,
            "correlation_y": 0.0,
            "correlation_z": 0.0,
            "correlation_mean": 0.0,
            "success_rate": 0.0,
            "n_reconstructed": 0,
            "n_total": n_total,
            "psnr_db": 0.0,
        }

    gt_valid = gt_3d[valid]
    recon_valid = recon_3d[valid]

    # 3D position errors
    errors_3d = np.linalg.norm(gt_valid - recon_valid, axis=1)
    rmse = np.sqrt(np.mean(errors_3d ** 2))
    mean_err = np.mean(errors_3d)
    median_err = np.median(errors_3d)
    max_err = np.max(errors_3d)

    # Per-axis correlation coefficients
    cc_x = np.corrcoef(gt_valid[:, 0], recon_valid[:, 0])[0, 1]
    cc_y = np.corrcoef(gt_valid[:, 1], recon_valid[:, 1])[0, 1]
    cc_z = np.corrcoef(gt_valid[:, 2], recon_valid[:, 2])[0, 1]
    cc_mean = (cc_x + cc_y + cc_z) / 3.0

    # Success rate
    success_rate = n_valid / n_total

    # PSNR: treat volume diagonal as signal range
    vol_diag = np.linalg.norm(VOL_MAX - VOL_MIN)
    mse = np.mean(errors_3d ** 2)
    psnr = 10.0 * np.log10(vol_diag ** 2 / mse) if mse > 0 else float('inf')

    return {
        "rmse_3d_mm": float(np.round(rmse, 4)),
        "mean_error_mm": float(np.round(mean_err, 4)),
        "median_error_mm": float(np.round(median_err, 4)),
        "max_error_mm": float(np.round(max_err, 4)),
        "correlation_x": float(np.round(cc_x, 6)),
        "correlation_y": float(np.round(cc_y, 6)),
        "correlation_z": float(np.round(cc_z, 6)),
        "correlation_mean": float(np.round(cc_mean, 6)),
        "success_rate": float(np.round(success_rate, 4)),
        "n_reconstructed": int(n_valid),
        "n_total": int(n_total),
        "psnr_db": float(np.round(psnr, 2)),
    }


# ═══════════════════════════════════════════════════════════
# 7. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(gt_3d, recon_3d, n_views, reproj_errors,
                      cameras, metrics, save_path):
    """
    Create comprehensive visualization:
      (a) 3D scatter: GT vs Reconstructed positions
      (b) Per-axis correlation plots (X, Y, Z)
      (c) 3D position error histogram
      (d) Camera layout + particle cloud
    """
    valid = ~np.isnan(recon_3d[:, 0])
    gt_v = gt_3d[valid]
    rc_v = recon_3d[valid]
    errors_3d = np.linalg.norm(gt_v - rc_v, axis=1)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "3D Particle Tracking Velocimetry — Multi-Camera Triangulation\n"
        f"RMSE={metrics['rmse_3d_mm']:.3f} mm  |  "
        f"CC={metrics['correlation_mean']:.4f}  |  "
        f"PSNR={metrics['psnr_db']:.1f} dB  |  "
        f"Success={metrics['success_rate']*100:.1f}%",
        fontsize=14, fontweight='bold'
    )

    # (a) 3D scatter: GT (blue) vs Recon (red)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2],
                c='blue', s=8, alpha=0.5, label='Ground Truth')
    ax1.scatter(rc_v[:, 0], rc_v[:, 1], rc_v[:, 2],
                c='red', s=8, alpha=0.5, label='Reconstructed')
    # Draw camera positions
    for cam in cameras:
        cp = cam['cam_pos']
        ax1.scatter(*cp, c='green', s=100, marker='^', zorder=5)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    ax1.set_title('3D Positions: GT vs Recon')
    ax1.legend(fontsize=8)

    # (b) X-axis correlation
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(gt_v[:, 0], rc_v[:, 0], s=10, alpha=0.6, c='steelblue')
    lim = [VOL_MIN[0] - 5, VOL_MAX[0] + 5]
    ax2.plot(lim, lim, 'k--', lw=1)
    ax2.set_xlabel('GT X [mm]')
    ax2.set_ylabel('Recon X [mm]')
    ax2.set_title(f'X Correlation (CC={metrics["correlation_x"]:.4f})')
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # (c) Y-axis correlation
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(gt_v[:, 1], rc_v[:, 1], s=10, alpha=0.6, c='coral')
    lim_y = [VOL_MIN[1] - 5, VOL_MAX[1] + 5]
    ax3.plot(lim_y, lim_y, 'k--', lw=1)
    ax3.set_xlabel('GT Y [mm]')
    ax3.set_ylabel('Recon Y [mm]')
    ax3.set_title(f'Y Correlation (CC={metrics["correlation_y"]:.4f})')
    ax3.set_xlim(lim_y)
    ax3.set_ylim(lim_y)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # (d) Z-axis correlation
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(gt_v[:, 2], rc_v[:, 2], s=10, alpha=0.6, c='green')
    lim_z = [VOL_MIN[2] - 5, VOL_MAX[2] + 5]
    ax4.plot(lim_z, lim_z, 'k--', lw=1)
    ax4.set_xlabel('GT Z [mm]')
    ax4.set_ylabel('Recon Z [mm]')
    ax4.set_title(f'Z Correlation (CC={metrics["correlation_z"]:.4f})')
    ax4.set_xlim(lim_z)
    ax4.set_ylim(lim_z)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # (e) 3D error histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(errors_3d, bins=40, color='steelblue', edgecolor='black',
             alpha=0.7)
    ax5.axvline(metrics['rmse_3d_mm'], color='red', ls='--',
                label=f'RMSE={metrics["rmse_3d_mm"]:.3f} mm')
    ax5.axvline(metrics['median_error_mm'], color='orange', ls='--',
                label=f'Median={metrics["median_error_mm"]:.3f} mm')
    ax5.set_xlabel('3D Position Error [mm]')
    ax5.set_ylabel('Count')
    ax5.set_title('Error Distribution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # (f) Number of views histogram
    ax6 = fig.add_subplot(2, 3, 6)
    view_counts = n_views[valid]
    bins_v = np.arange(0.5, N_CAMERAS + 1.5, 1)
    ax6.hist(view_counts, bins=bins_v, color='mediumpurple',
             edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Number of Camera Views')
    ax6.set_ylabel('Count')
    ax6.set_title('Views per Particle')
    ax6.set_xticks(range(1, N_CAMERAS + 1))
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  myptv — 3D Particle Tracking Velocimetry")
    print("=" * 60)

    # (a) Setup cameras
    print("\n[SETUP] Configuring multi-camera system...")
    cameras = setup_cameras()
    for cam in cameras:
        print(f"  Camera {cam['id']}: pos={cam['cam_pos'].round(1)} mm")

    # (b) Generate ground truth particles
    print(f"\n[DATA] Generating {N_PARTICLES} particles in volume "
          f"{VOL_MIN} → {VOL_MAX} mm...")
    particles_gt = generate_particles()
    print(f"[DATA] Shape: {particles_gt.shape}")

    # (c) Forward: project 3D → 2D on all cameras
    print(f"\n[FWD] Projecting particles onto {N_CAMERAS} cameras "
          f"(noise σ={NOISE_STD_PX} px)...")
    detections, visibility, clean_proj = forward_operator(
        particles_gt, cameras, noise_std=NOISE_STD_PX
    )
    for i in range(N_CAMERAS):
        n_vis = np.sum(visibility[i])
        print(f"  Camera {i}: {n_vis}/{N_PARTICLES} particles visible")

    # (d) Inverse: triangulate 3D from 2D detections
    print(f"\n[RECON] Triangulating 3D positions via DLT...")
    recon_3d, n_views, reproj_errors = inverse_operator(
        detections, visibility, cameras
    )
    n_recon = np.sum(~np.isnan(recon_3d[:, 0]))
    print(f"[RECON] Reconstructed {n_recon}/{N_PARTICLES} particles")
    valid_reproj = reproj_errors[~np.isnan(reproj_errors)]
    if len(valid_reproj) > 0:
        print(f"[RECON] Reprojection error: "
              f"mean={np.mean(valid_reproj):.3f} px, "
              f"max={np.max(valid_reproj):.3f} px")

    # (e) Compute metrics
    print("\n[EVAL] Computing metrics...")
    metrics = compute_metrics(particles_gt, recon_3d, n_views)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")
        else:
            print(f"  {k} = {v}")

    # (f) Save results
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")

    # Save ground truth and reconstruction as .npy
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), particles_gt)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_3d)

    # Also save input (2D detections stacked)
    input_data = {
        'detections': [d.tolist() for d in detections],
        'visibility': [v.tolist() for v in visibility],
    }
    np.save(os.path.join(RESULTS_DIR, "input.npy"),
            np.array(input_data, dtype=object))

    print(f"[SAVE] GT shape: {particles_gt.shape}")
    print(f"[SAVE] Recon shape: {recon_3d.shape}")

    # (g) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    print(f"\n[VIS] Generating visualization...")
    visualize_results(
        particles_gt, recon_3d, n_views, reproj_errors,
        cameras, metrics, vis_path
    )

    print("\n" + "=" * 60)
    print("  DONE — myptv 3D PTV Reconstruction")
    print("=" * 60)
