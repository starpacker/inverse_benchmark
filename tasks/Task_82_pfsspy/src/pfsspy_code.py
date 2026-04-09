"""
pfsspy — Potential Field Source Surface (PFSS) Coronal Magnetic Field
======================================================================
Task: Reconstruct 3D coronal magnetic field from photospheric magnetogram
Repo: https://github.com/dstansby/pfsspy
Paper: Stansby et al., "pfsspy: A Python package for potential field
       source surface modelling" (JOSS, 2020)

Inverse Problem:
    Forward: Given 3D magnetic field B(r,θ,φ), extract the photospheric
             boundary condition B_r(R_sun, θ, φ)
    Inverse: From photospheric magnetogram B_r(R_sun), solve Laplace's equation
             ∇²Ψ = 0 with boundary conditions at R_sun and R_ss (source surface)
             to recover B(r,θ,φ) = -∇Ψ in the corona

Usage:
    /data/yjh/pfsspy_env/bin/python pfsspy_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# PFSS parameters
NPHI = 72         # Number of longitude points
NTHETA = 36       # Number of latitude points (sin(lat) grid)
NR = 30           # Number of radial grid points
RSS = 2.5         # Source surface radius (in solar radii)
NOISE_LEVEL = 0.05  # Noise level for magnetogram (fraction of max)


# ═══════════════════════════════════════════════════════════
# 2. Synthetic Magnetogram Generation
# ═══════════════════════════════════════════════════════════
def generate_synthetic_magnetogram():
    """
    Generate a synthetic photospheric magnetogram with dipole + quadrupole
    components, simulating a simplified solar magnetic field.
    
    Returns:
        br_map: sunpy Map of radial magnetic field at photosphere
        br_clean: clean magnetogram (without noise)
        coefficients: spherical harmonic coefficients (ground truth)
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map
    
    # Create coordinate grid
    # For CEA projection, y-axis is proportional to sin(latitude)
    phi = np.linspace(0, 2 * np.pi, NPHI + 1)[:-1]  # longitude
    sin_lat = np.linspace(-1, 1, NTHETA)  # sin(latitude) for CEA
    theta = np.arcsin(sin_lat)  # latitude in radians
    
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Synthesize magnetogram using spherical harmonics
    # Dipole component (l=1, m=0): B_r ~ cos(θ) (tilted slightly)
    dipole_strength = 5.0  # Gauss
    br_dipole = dipole_strength * np.sin(theta_grid)  # axial dipole
    
    # Tilted dipole component (l=1, m=1)
    tilt_strength = 2.0
    br_tilt = tilt_strength * np.cos(theta_grid) * np.cos(phi_grid - 0.5)
    
    # Quadrupole component (l=2, m=0)
    quad_strength = 1.5
    br_quad = quad_strength * (3 * np.sin(theta_grid)**2 - 1) / 2
    
    # Active region spots (localized bipolar regions)
    # Spot 1: positive polarity
    lat1, lon1 = np.deg2rad(20), np.deg2rad(60)
    sigma_spot = np.deg2rad(10)
    dist1 = np.sqrt((theta_grid - lat1)**2 + (phi_grid - lon1)**2)
    br_spot1 = 15.0 * np.exp(-dist1**2 / (2 * sigma_spot**2))
    
    # Spot 2: negative polarity (nearby)
    lat2, lon2 = np.deg2rad(25), np.deg2rad(75)
    dist2 = np.sqrt((theta_grid - lat2)**2 + (phi_grid - lon2)**2)
    br_spot2 = -12.0 * np.exp(-dist2**2 / (2 * sigma_spot**2))
    
    # Another active region in southern hemisphere
    lat3, lon3 = np.deg2rad(-15), np.deg2rad(200)
    dist3 = np.sqrt((theta_grid - lat3)**2 + (phi_grid - lon3)**2)
    br_spot3 = -10.0 * np.exp(-dist3**2 / (2 * sigma_spot**2))
    
    lat4, lon4 = np.deg2rad(-10), np.deg2rad(220)
    dist4 = np.sqrt((theta_grid - lat4)**2 + (phi_grid - lon4)**2)
    br_spot4 = 8.0 * np.exp(-dist4**2 / (2 * sigma_spot**2))
    
    # Combined clean magnetogram
    br_clean = br_dipole + br_tilt + br_quad + br_spot1 + br_spot2 + br_spot3 + br_spot4
    
    # Add noise
    noise = NOISE_LEVEL * br_clean.max() * np.random.randn(*br_clean.shape)
    br_noisy = br_clean + noise
    
    # Create a SunPy Map with CEA projection (required by pfsspy)
    # CEA: Cylindrical Equal Area
    # pfsspy validation: shape[1]*CDELT1 ≈ 360°, shape[0]*CDELT2*π/2 ≈ 180°
    # So CDELT2 = 360 / (NTHETA * π)
    cdelt1 = 360.0 / NPHI
    cdelt2 = 360.0 / (NTHETA * np.pi)
    header = {
        'NAXIS1': NPHI,
        'NAXIS2': NTHETA,
        'CDELT1': cdelt1,
        'CDELT2': cdelt2,
        'CRPIX1': (NPHI + 1) / 2.0,
        'CRPIX2': (NTHETA + 1) / 2.0,
        'CRVAL1': 0.0,
        'CRVAL2': 0.0,
        'CTYPE1': 'CRLN-CEA',
        'CTYPE2': 'CRLT-CEA',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'DATE-OBS': '2024-01-01T00:00:00',
        'BUNIT': 'G',
    }
    br_map = sunpy.map.Map(br_noisy, header)
    
    return br_map, br_clean, br_noisy


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator
# ═══════════════════════════════════════════════════════════
def forward_operator(output):
    """
    Forward: Extract photospheric B_r from PFSS solution.
    
    Given the 3D magnetic field solution B(r,θ,φ), extract the 
    radial component at the photosphere (r = R_sun).
    """
    # Get the magnetic field at the inner boundary
    br_photosphere = output.bc[0][:, :, 0]  # B_r at inner boundary
    return br_photosphere


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: PFSS
# ═══════════════════════════════════════════════════════════
def reconstruct(br_map):
    """
    PFSS reconstruction: solve Laplace's equation for the coronal field.
    
    Solves ∇²Ψ = 0 with boundary conditions:
    - Inner boundary (r = R_sun): B_r = -∂Ψ/∂r from magnetogram
    - Outer boundary (r = R_ss): B_r = 0 (purely radial at source surface)
    
    Solution via spherical harmonic decomposition.
    """
    import pfsspy
    
    print(f"  [PFSS] Creating input: nr={NR}, rss={RSS}")
    pfss_input = pfsspy.Input(br_map, NR, RSS)
    
    print("  [PFSS] Solving PFSS equations...")
    output = pfsspy.pfss(pfss_input)
    
    print(f"  [PFSS] Solution grid: {output.bg.shape}")
    print(f"  [PFSS] Source surface radius: {RSS} R_sun")
    
    return output


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(br_clean, output):
    """
    Evaluate PFSS reconstruction quality.
    
    Compare the reconstructed photospheric B_r (from PFSS solution)
    with the clean (noise-free) input magnetogram.
    Also evaluate field properties at different heights.
    """
    # Get magnetic field components from PFSS output
    bg_raw = output.bg  # (nphi+1, ns+1, nr+1, 3) - B field on cell boundaries
    bg = np.array(bg_raw.value if hasattr(bg_raw, 'value') else bg_raw)
    
    # Extract B_r at photosphere from reconstruction  
    # The output stores field on grid cell faces; .value strips astropy units
    # Note: output.bc is in (nphi, ntheta, nr) order, transpose to (ntheta, nphi)
    br_recon_raw = output.bc[0][:, :, 0]
    br_recon = np.array(br_recon_raw.value if hasattr(br_recon_raw, 'value') else br_recon_raw).T
    
    # Resize clean to match recon shape if needed
    from scipy.ndimage import zoom
    if br_clean.shape != br_recon.shape:
        zoom_factors = [br_recon.shape[i] / br_clean.shape[i] for i in range(2)]
        br_clean_resized = zoom(br_clean, zoom_factors)
    else:
        br_clean_resized = br_clean
    
    # Flatten for comparison
    gt = br_clean_resized.flatten()
    recon = br_recon.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((gt - recon)**2))
    
    # Correlation coefficient
    cc = np.corrcoef(gt, recon)[0, 1]
    
    # Relative error
    re = np.sqrt(np.mean((gt - recon)**2)) / np.sqrt(np.mean(gt**2))
    
    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - recon)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # Magnetic energy (proxy for field quality)
    # Total unsigned flux at photosphere
    total_flux_gt = np.sum(np.abs(gt))
    total_flux_recon = np.sum(np.abs(recon))
    flux_ratio = total_flux_recon / total_flux_gt if total_flux_gt > 0 else 0
    
    # B_r at source surface (should be ~0 for open field)
    br_ss_raw = output.bc[0][:, :, -1]
    br_ss = np.array(br_ss_raw.value if hasattr(br_ss_raw, 'value') else br_ss_raw).T
    max_br_ss = np.max(np.abs(br_ss))
    
    # Open flux
    open_flux = np.sum(np.abs(br_ss))
    
    return {
        'psnr': float(psnr),
        'rmse': float(rmse),
        'cc': float(cc),
        'relative_error': float(re),
        'flux_ratio': float(flux_ratio),
        'max_br_source_surface': float(max_br_ss),
        'open_flux': float(open_flux),
        'br_recon_shape': list(br_recon.shape),
        'bg_shape': list(bg.shape),
    }


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(br_clean, br_noisy, output, metrics, save_path):
    """Generate comprehensive PFSS visualization."""
    # Get reconstructed B_r at photosphere (strip units)
    # Note: output.bc is in (nphi, ntheta, nr) order, transpose to (ntheta, nphi)
    br_recon_raw = output.bc[0][:, :, 0]
    br_recon = np.array(br_recon_raw.value if hasattr(br_recon_raw, 'value') else br_recon_raw).T
    
    # Resize clean for comparison
    from scipy.ndimage import zoom
    if br_clean.shape != br_recon.shape:
        zoom_factors = [br_recon.shape[i] / br_clean.shape[i] for i in range(2)]
        br_clean_r = zoom(br_clean, zoom_factors)
        br_noisy_r = zoom(br_noisy, zoom_factors)
    else:
        br_clean_r = br_clean
        br_noisy_r = br_noisy
    
    vmax = max(np.abs(br_clean_r).max(), np.abs(br_recon).max())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Clean magnetogram
    im0 = axes[0, 0].imshow(br_clean_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 0].set_title('GT Magnetogram (clean)')
    axes[0, 0].set_xlabel('Longitude (px)')
    axes[0, 0].set_ylabel('Sine Latitude (px)')
    plt.colorbar(im0, ax=axes[0, 0], label='B_r (G)')
    
    # (b) Noisy magnetogram (input)
    im1 = axes[0, 1].imshow(br_noisy_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 1].set_title('Input Magnetogram (noisy)')
    axes[0, 1].set_xlabel('Longitude (px)')
    plt.colorbar(im1, ax=axes[0, 1], label='B_r (G)')
    
    # (c) Reconstructed B_r at photosphere
    im2 = axes[0, 2].imshow(br_recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 2].set_title('PFSS Reconstructed B_r')
    axes[0, 2].set_xlabel('Longitude (px)')
    plt.colorbar(im2, ax=axes[0, 2], label='B_r (G)')
    
    # (d) Error map
    error = br_clean_r - br_recon
    im3 = axes[1, 0].imshow(error, cmap='seismic', 
                             vmin=-vmax*0.3, vmax=vmax*0.3,
                             aspect='auto', origin='lower')
    axes[1, 0].set_title('Error (GT - Recon)')
    axes[1, 0].set_xlabel('Longitude (px)')
    axes[1, 0].set_ylabel('Sine Latitude (px)')
    plt.colorbar(im3, ax=axes[1, 0], label='ΔB_r (G)')
    
    # (e) B_r at source surface
    br_ss_raw = output.bc[0][:, :, -1]
    br_ss = np.array(br_ss_raw.value if hasattr(br_ss_raw, 'value') else br_ss_raw).T
    im4 = axes[1, 1].imshow(br_ss, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1, 1].set_title(f'B_r at Source Surface (R={RSS} R_sun)')
    axes[1, 1].set_xlabel('Longitude (px)')
    plt.colorbar(im4, ax=axes[1, 1], label='B_r (G)')
    
    # (f) Scatter: GT vs Recon
    axes[1, 2].scatter(br_clean_r.flatten(), br_recon.flatten(), 
                       alpha=0.3, s=5, c='steelblue')
    lim = vmax * 1.1
    axes[1, 2].plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Identity')
    axes[1, 2].set_xlabel('GT B_r (G)')
    axes[1, 2].set_ylabel('Recon B_r (G)')
    axes[1, 2].set_title(f'GT vs Recon (CC={metrics["cc"]:.4f})')
    axes[1, 2].legend()
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_xlim([-lim, lim])
    axes[1, 2].set_ylim([-lim, lim])
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"pfsspy — PFSS Coronal Magnetic Field Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC={metrics['cc']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} G | RE={metrics['relative_error']:.4f}",
        fontsize=12, fontweight='bold'
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
    print("  pfsspy — PFSS Coronal Magnetic Field Reconstruction")
    print("=" * 60)
    
    # (a) Generate synthetic magnetogram
    print("\n[DATA] Generating synthetic photospheric magnetogram...")
    br_map, br_clean, br_noisy = generate_synthetic_magnetogram()
    print(f"[DATA] Magnetogram shape: {br_clean.shape}")
    print(f"[DATA] B_r range: [{br_clean.min():.2f}, {br_clean.max():.2f}] G")
    
    # (b) Run PFSS reconstruction
    print("\n[RECON] Running PFSS reconstruction...")
    output = reconstruct(br_map)
    
    # (c) Evaluate
    print("\n[EVAL] Computing evaluation metrics...")
    metrics = compute_metrics(br_clean, output)
    
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMSE = {metrics['rmse']:.4f} G")
    print(f"[EVAL] Relative Error = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Flux ratio = {metrics['flux_ratio']:.4f}")
    print(f"[EVAL] Max B_r at source surface = {metrics['max_br_source_surface']:.4f} G")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Save arrays
    br_recon_raw = output.bc[0][:, :, 0]
    br_recon = np.array(br_recon_raw.value if hasattr(br_recon_raw, 'value') else br_recon_raw).T
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), br_clean)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), br_recon)
    np.save(os.path.join(RESULTS_DIR, "input.npy"), br_noisy)
    print(f"[SAVE] GT shape: {br_clean.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {br_recon.shape} → reconstruction.npy")
    print(f"[SAVE] Input shape: {br_noisy.shape} → input.npy")
    
    # (f) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(br_clean, br_noisy, output, metrics, vis_path)
    
    print("\n" + "=" * 60)
    print("  DONE — pfsspy PFSS Reconstruction")
    print("=" * 60)
