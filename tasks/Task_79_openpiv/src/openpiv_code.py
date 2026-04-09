"""
OpenPIV - Particle Image Velocimetry (PIV) Flow Reconstruction
================================================================
Task: Recover 2D velocity field from synthetic particle image pairs
Repo: https://github.com/OpenPIV/openpiv-python
Paper: Taylor et al., "OpenPIV: A Python Toolbox for Particle Image Velocimetry"

Inverse Problem:
    Forward: A known velocity field u(x,y), v(x,y) displaces tracer particles
             between two camera frames (frame_a -> frame_b)
    Inverse: Given frame pair (frame_a, frame_b), recover the velocity field
             u_hat(x,y), v_hat(x,y) via cross-correlation analysis

Usage:
    /data/yjh/openpiv_env/bin/python openpiv_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# PIV parameters
IMG_SIZE = 512           # Image size in pixels
N_PARTICLES = 8000       # Number of tracer particles
PARTICLE_DIAMETER = 3.0  # Particle diameter in pixels
WINDOW_SIZE = 32         # Interrogation window size
OVERLAP = 16             # Overlap between windows
SEARCH_AREA_SIZE = 64    # Search area size (for extended search)
NOISE_LEVEL = 0.02       # Image noise level (fraction of max intensity)
DT = 1.0                 # Time between frames

np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# 2. Synthetic Data Generation (Forward Model)
# ═══════════════════════════════════════════════════════════
def generate_gaussian_particle(img, x0, y0, diameter, intensity=255.0):
    """Render a single Gaussian particle onto the image."""
    sigma = diameter / 4.0
    r = int(3 * sigma) + 1
    y_min = max(0, int(y0) - r)
    y_max = min(img.shape[0], int(y0) + r + 1)
    x_min = max(0, int(x0) - r)
    x_max = min(img.shape[1], int(x0) + r + 1)
    
    for iy in range(y_min, y_max):
        for ix in range(x_min, x_max):
            dist2 = (ix - x0)**2 + (iy - y0)**2
            img[iy, ix] += intensity * np.exp(-dist2 / (2 * sigma**2))
    return img


def generate_velocity_field_on_grid(grid_x, grid_y):
    """
    Generate ground truth velocity field on a given grid.
    Uses a combination of:
    - Lamb-Oseen vortex (rotational flow)
    - Uniform translation
    - Strain field
    """
    # Normalize coordinates to [0, 1]
    cx, cy = IMG_SIZE / 2.0, IMG_SIZE / 2.0
    
    # Lamb-Oseen vortex parameters
    vortex_strength = 5.0  # pixels/frame
    vortex_radius = IMG_SIZE / 6.0
    
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-10
    
    # Tangential velocity (Lamb-Oseen vortex)
    v_theta = vortex_strength * (1 - np.exp(-r**2 / (2 * vortex_radius**2))) / r
    u_vortex = -v_theta * dy  # u = -v_theta * sin(theta)
    v_vortex = v_theta * dx   # v =  v_theta * cos(theta)
    
    # Add uniform translation
    u_uniform = 2.0  # pixels/frame
    v_uniform = 1.0
    
    # Add strain field (convergent/divergent)
    strain_rate = 0.005
    u_strain = strain_rate * dx
    v_strain = -strain_rate * dy
    
    u_total = u_vortex + u_uniform + u_strain
    v_total = v_vortex + v_uniform + v_strain
    
    return u_total, v_total


def generate_particle_images():
    """
    Forward operator: generate synthetic particle image pair from known velocity field.
    
    Returns:
        frame_a: first frame (2D array)
        frame_b: second frame (particles displaced by velocity field)
        gt_u: ground truth u velocity on PIV grid
        gt_v: ground truth v velocity on PIV grid
        particle_positions: (x, y) positions of particles in frame_a
    """
    # Generate random particle positions for frame_a
    x_particles = np.random.uniform(10, IMG_SIZE - 10, N_PARTICLES)
    y_particles = np.random.uniform(10, IMG_SIZE - 10, N_PARTICLES)
    
    # Compute velocity at each particle position
    u_at_particles, v_at_particles = generate_velocity_field_on_grid(
        x_particles, y_particles
    )
    
    # Displaced positions for frame_b
    x_displaced = x_particles + u_at_particles * DT
    y_displaced = y_particles + v_at_particles * DT
    
    # Random intensities for particles
    intensities = np.random.uniform(180, 255, N_PARTICLES)
    
    # Render frame_a
    frame_a = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    for i in range(N_PARTICLES):
        frame_a = generate_gaussian_particle(
            frame_a, x_particles[i], y_particles[i], 
            PARTICLE_DIAMETER, intensities[i]
        )
    
    # Render frame_b (particles displaced by velocity field)
    frame_b = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    for i in range(N_PARTICLES):
        # Only render if displaced position is within bounds
        if (5 < x_displaced[i] < IMG_SIZE - 5 and 
            5 < y_displaced[i] < IMG_SIZE - 5):
            frame_b = generate_gaussian_particle(
                frame_b, x_displaced[i], y_displaced[i],
                PARTICLE_DIAMETER, intensities[i]
            )
    
    # Clip to valid range and add noise
    frame_a = np.clip(frame_a, 0, 255)
    frame_b = np.clip(frame_b, 0, 255)
    
    # Add Gaussian noise
    noise_a = np.random.normal(0, NOISE_LEVEL * 255, frame_a.shape)
    noise_b = np.random.normal(0, NOISE_LEVEL * 255, frame_b.shape)
    frame_a = np.clip(frame_a + noise_a, 0, 255).astype(np.int32)
    frame_b = np.clip(frame_b + noise_b, 0, 255).astype(np.int32)
    
    # Compute ground truth on PIV grid
    from openpiv.pyprocess import get_coordinates
    x_grid, y_grid = get_coordinates(
        image_size=frame_a.shape, 
        search_area_size=SEARCH_AREA_SIZE, 
        overlap=OVERLAP
    )
    gt_u, gt_v = generate_velocity_field_on_grid(x_grid, y_grid)
    
    return frame_a, frame_b, gt_u, gt_v, x_grid, y_grid


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator
# ═══════════════════════════════════════════════════════════
def forward_operator(velocity_field_u, velocity_field_v, particle_positions_x, particle_positions_y):
    """
    Forward model: Given velocity field, displace particles to create frame_b.
    y = A(x): velocity field -> displaced particle image
    
    In PIV, the forward model is:
        x'_i = x_i + u(x_i, y_i) * dt
        y'_i = y_i + v(x_i, y_i) * dt
    where (x_i, y_i) are particle positions and u, v is the velocity field.
    """
    u_at_particles, v_at_particles = generate_velocity_field_on_grid(
        particle_positions_x, particle_positions_y
    )
    x_displaced = particle_positions_x + u_at_particles * DT
    y_displaced = particle_positions_y + v_at_particles * DT
    return x_displaced, y_displaced


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: PIV Cross-Correlation
# ═══════════════════════════════════════════════════════════
def reconstruct(frame_a, frame_b):
    """
    PIV reconstruction: recover velocity field from particle image pair.
    
    Uses OpenPIV's extended search area PIV with:
    1. Cross-correlation in interrogation windows
    2. Sub-pixel peak finding (Gaussian fit)
    3. Signal-to-noise ratio filtering
    4. Outlier replacement via local mean interpolation
    
    Returns:
        u_recon: reconstructed u velocity field
        v_recon: reconstructed v velocity field
        x_grid: x coordinates of PIV grid
        y_grid: y coordinates of PIV grid
    """
    from openpiv import pyprocess, validation, filters
    
    # Step 1: Cross-correlation PIV
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        dt=DT,
        search_area_size=SEARCH_AREA_SIZE,
        correlation_method='circular',
        subpixel_method='gaussian',
        sig2noise_method='peak2peak'
    )
    
    # Step 2: Get coordinates
    x_grid, y_grid = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=SEARCH_AREA_SIZE,
        overlap=OVERLAP
    )
    
    # Step 3: Validation - signal-to-noise ratio filter
    flags_s2n = validation.sig2noise_val(sig2noise, threshold=1.05)
    
    # Step 4: Validation - global velocity range filter
    u_max = np.max(np.abs(u)) * 1.5 + 1
    flags_g = validation.global_val(u, v, (-u_max, u_max), (-u_max, u_max))
    
    # Combine flags
    flags = flags_s2n | flags_g
    
    # Step 5: Replace outliers using local mean interpolation
    u_filtered, v_filtered = filters.replace_outliers(
        u, v, flags, 
        method='localmean', 
        max_iter=10, 
        kernel_size=2
    )
    
    # Convert MaskedArray to regular ndarray (openpiv may return MaskedArray)
    if hasattr(u_filtered, 'data'):
        u_filtered = np.array(u_filtered)
    if hasattr(v_filtered, 'data'):
        v_filtered = np.array(v_filtered)
    
    print(f"  [PIV] Grid shape: {u_filtered.shape}")
    print(f"  [PIV] Outliers replaced: {np.sum(flags)} / {flags.size} "
          f"({100*np.sum(flags)/flags.size:.1f}%)")
    print(f"  [PIV] Velocity range: u=[{u_filtered.min():.2f}, {u_filtered.max():.2f}], "
          f"v=[{v_filtered.min():.2f}, {v_filtered.max():.2f}]")
    
    return u_filtered, v_filtered, x_grid, y_grid


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_velocity_metrics(gt_u, gt_v, recon_u, recon_v):
    """
    Compute PIV-specific evaluation metrics.
    
    Returns dict with:
    - rmse_u, rmse_v: RMSE for each component
    - rmse_magnitude: RMSE for velocity magnitude
    - cc_u, cc_v: Correlation coefficients
    - re: Relative error (||error|| / ||gt||)
    - psnr: PSNR based on velocity magnitude range
    - aee: Average Endpoint Error
    """
    # Velocity magnitudes
    gt_mag = np.sqrt(gt_u**2 + gt_v**2)
    recon_mag = np.sqrt(recon_u**2 + recon_v**2)
    
    # RMSE per component
    rmse_u = np.sqrt(np.mean((gt_u - recon_u)**2))
    rmse_v = np.sqrt(np.mean((gt_v - recon_v)**2))
    
    # RMSE of magnitude
    rmse_mag = np.sqrt(np.mean((gt_mag - recon_mag)**2))
    
    # Correlation coefficient
    cc_u = np.corrcoef(gt_u.flatten(), recon_u.flatten())[0, 1]
    cc_v = np.corrcoef(gt_v.flatten(), recon_v.flatten())[0, 1]
    cc_mag = np.corrcoef(gt_mag.flatten(), recon_mag.flatten())[0, 1]
    
    # Relative Error
    gt_norm = np.sqrt(np.mean(gt_u**2 + gt_v**2))
    error_norm = np.sqrt(np.mean((gt_u - recon_u)**2 + (gt_v - recon_v)**2))
    re = error_norm / gt_norm if gt_norm > 0 else float('inf')
    
    # Average Endpoint Error (standard PIV metric)
    aee = np.mean(np.sqrt((gt_u - recon_u)**2 + (gt_v - recon_v)**2))
    
    # PSNR based on velocity magnitude
    data_range = gt_mag.max() - gt_mag.min()
    mse = np.mean((gt_mag - recon_mag)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # SSIM of velocity magnitude (treated as 2D image)
    from skimage.metrics import structural_similarity as ssim
    gt_mag_norm = (gt_mag - gt_mag.min()) / (gt_mag.max() - gt_mag.min() + 1e-10)
    recon_mag_norm = (recon_mag - recon_mag.min()) / (recon_mag.max() - recon_mag.min() + 1e-10)
    # Handle small images where default win_size is too large
    min_dim = min(gt_mag_norm.shape)
    win_size = min(7, min_dim) if min_dim >= 3 else 3
    if win_size % 2 == 0:
        win_size -= 1
    ssim_val = ssim(gt_mag_norm, recon_mag_norm, data_range=1.0, win_size=win_size)
    
    return {
        'rmse_u': float(rmse_u),
        'rmse_v': float(rmse_v),
        'rmse_magnitude': float(rmse_mag),
        'cc_u': float(cc_u),
        'cc_v': float(cc_v),
        'cc_magnitude': float(cc_mag),
        'relative_error': float(re),
        'average_endpoint_error': float(aee),
        'psnr': float(psnr),
        'ssim': float(ssim_val),
        'rmse': float(rmse_mag),  # Alias for standard metrics
    }


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(frame_a, frame_b, gt_u, gt_v, recon_u, recon_v, 
                      x_grid, y_grid, metrics, save_path):
    """Generate comprehensive PIV visualization."""
    gt_mag = np.sqrt(gt_u**2 + gt_v**2)
    recon_mag = np.sqrt(recon_u**2 + recon_v**2)
    error_mag = np.abs(gt_mag - recon_mag)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Images and velocity fields
    # (a) Frame A
    axes[0, 0].imshow(frame_a, cmap='gray')
    axes[0, 0].set_title('Frame A (Particle Image)', fontsize=12)
    axes[0, 0].axis('off')
    
    # (b) Frame B
    axes[0, 1].imshow(frame_b, cmap='gray')
    axes[0, 1].set_title('Frame B (Displaced Particles)', fontsize=12)
    axes[0, 1].axis('off')
    
    # (c) GT velocity magnitude
    im_gt = axes[0, 2].imshow(gt_mag, cmap='jet', origin='upper')
    axes[0, 2].set_title('Ground Truth |V|', fontsize=12)
    plt.colorbar(im_gt, ax=axes[0, 2], fraction=0.046, label='pixels/frame')
    
    # (d) Reconstructed velocity magnitude
    im_recon = axes[0, 3].imshow(recon_mag, cmap='jet', origin='upper',
                                  vmin=gt_mag.min(), vmax=gt_mag.max())
    axes[0, 3].set_title('Reconstructed |V|', fontsize=12)
    plt.colorbar(im_recon, ax=axes[0, 3], fraction=0.046, label='pixels/frame')
    
    # Row 2: Quiver plots and error
    # (e) GT quiver
    skip = 1  # downsample for clarity
    axes[1, 0].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                       gt_u[::skip, ::skip], -gt_v[::skip, ::skip],
                       gt_mag[::skip, ::skip], cmap='jet',
                       scale=None, width=0.004)
    axes[1, 0].set_title('GT Velocity Field', fontsize=12)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()
    
    # (f) Reconstructed quiver
    axes[1, 1].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                       recon_u[::skip, ::skip], -recon_v[::skip, ::skip],
                       recon_mag[::skip, ::skip], cmap='jet',
                       scale=None, width=0.004)
    axes[1, 1].set_title('Reconstructed Velocity Field', fontsize=12)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    
    # (g) Error map
    im_err = axes[1, 2].imshow(error_mag, cmap='hot', origin='upper')
    axes[1, 2].set_title('Velocity Magnitude Error', fontsize=12)
    plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, label='pixels/frame')
    
    # (h) Scatter plot: GT vs Reconstructed
    axes[1, 3].scatter(gt_mag.flatten(), recon_mag.flatten(), alpha=0.5, s=10, c='steelblue')
    max_val = max(gt_mag.max(), recon_mag.max()) * 1.1
    axes[1, 3].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Identity')
    axes[1, 3].set_xlabel('GT |V| (px/frame)', fontsize=11)
    axes[1, 3].set_ylabel('Recon |V| (px/frame)', fontsize=11)
    axes[1, 3].set_title(f'GT vs Recon (CC={metrics["cc_magnitude"]:.4f})', fontsize=12)
    axes[1, 3].legend()
    axes[1, 3].set_aspect('equal')
    axes[1, 3].set_xlim([0, max_val])
    axes[1, 3].set_ylim([0, max_val])
    
    fig.suptitle(
        f"OpenPIV — PIV Flow Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"AEE={metrics['average_endpoint_error']:.4f} px | "
        f"CC={metrics['cc_magnitude']:.4f} | RE={metrics['relative_error']:.4f}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  OpenPIV — PIV Flow Reconstruction")
    print("=" * 60)
    
    # (a) Generate synthetic data
    print("\n[DATA] Generating synthetic particle image pair...")
    frame_a, frame_b, gt_u, gt_v, x_grid, y_grid = generate_particle_images()
    print(f"[DATA] Frame shape: {frame_a.shape}")
    print(f"[DATA] GT velocity grid shape: {gt_u.shape}")
    print(f"[DATA] Particles: {N_PARTICLES}, Window: {WINDOW_SIZE}, "
          f"Overlap: {OVERLAP}, Search: {SEARCH_AREA_SIZE}")
    
    # (b) Run PIV reconstruction
    print("\n[RECON] Running PIV cross-correlation analysis...")
    recon_u, recon_v, rx_grid, ry_grid = reconstruct(frame_a, frame_b)
    print(f"[RECON] Reconstructed velocity grid shape: {recon_u.shape}")
    
    # (c) Evaluate
    print("\n[EVAL] Computing evaluation metrics...")
    metrics = compute_velocity_metrics(gt_u, gt_v, recon_u, recon_v)
    
    print(f"[EVAL] RMSE_u = {metrics['rmse_u']:.6f} px/frame")
    print(f"[EVAL] RMSE_v = {metrics['rmse_v']:.6f} px/frame")
    print(f"[EVAL] RMSE_|V| = {metrics['rmse_magnitude']:.6f} px/frame")
    print(f"[EVAL] CC_u = {metrics['cc_u']:.6f}")
    print(f"[EVAL] CC_v = {metrics['cc_v']:.6f}")
    print(f"[EVAL] CC_|V| = {metrics['cc_magnitude']:.6f}")
    print(f"[EVAL] Relative Error = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Average Endpoint Error = {metrics['average_endpoint_error']:.6f} px")
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] SSIM = {metrics['ssim']:.6f}")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Save arrays
    # Ground truth: stack u, v as (2, H, W)
    gt_velocity = np.stack([gt_u, gt_v], axis=0)
    recon_velocity = np.stack([recon_u, recon_v], axis=0)
    input_data = np.stack([frame_a, frame_b], axis=0)
    
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_velocity)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_velocity)
    np.save(os.path.join(RESULTS_DIR, "input.npy"), input_data)
    print(f"[SAVE] Ground truth shape: {gt_velocity.shape} → ground_truth.npy")
    print(f"[SAVE] Reconstruction shape: {recon_velocity.shape} → reconstruction.npy")
    print(f"[SAVE] Input shape: {input_data.shape} → input.npy")
    
    # (f) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(frame_a, frame_b, gt_u, gt_v, recon_u, recon_v,
                      x_grid, y_grid, metrics, vis_path)
    
    print("\n" + "=" * 60)
    print("  DONE — OpenPIV PIV Flow Reconstruction")
    print("=" * 60)
