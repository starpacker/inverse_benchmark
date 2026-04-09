"""
Task 163: harmonica_gravity — Gravity Field Inversion using Equivalent Sources

Inverse Problem: From surface gravity observations, infer subsurface density
distribution using equivalent sources (harmonica library).

Pipeline:
  1. Define subsurface density model (rectangular prisms)
  2. Forward: Compute gravity anomaly at surface using harmonica.prism_gravity
  3. Add Gaussian noise to simulate noisy observations
  4. Inverse: Fit harmonica.EquivalentSources to the noisy data
  5. Predict/reconstruct the gravity field from the equivalent source model
  6. Evaluate reconstruction quality (PSNR, SSIM, CC, RMSE)
  7. Save outputs and produce 4-panel visualization
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func, peak_signal_noise_ratio as psnr_func

import harmonica as hm
import verde as vd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SANDBOX = "/data/yjh/harmonica_gravity_sandbox"
ASSETS  = "/data/yjh/website_assets/Task_163_harmonica_gravity"
os.makedirs(SANDBOX, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Define subsurface density model (rectangular prisms)
# ──────────────────────────────────────────────────────────────────────────────
# Each prism: [west, east, south, north, bottom, top] in meters
# Positive density contrast = denser than surroundings
prisms = [
    [-3000, -500, -3000, -500, -5000, -1000],   # shallow dense block (NW)
    [1000, 4000, 1000, 4000, -8000, -3000],      # deeper block (SE)
    [-2000, 1000, 3000, 6000, -4000, -1500],     # medium block (NE)
    [4000, 7000, -6000, -3000, -6000, -2000],    # block (S)
]
densities = np.array([500.0, -400.0, 300.0, -250.0])  # kg/m³ anomalies

print(f"[INFO] Defined {len(prisms)} subsurface prisms")
for i, (p, d) in enumerate(zip(prisms, densities)):
    print(f"  Prism {i}: bounds={p}, density_contrast={d} kg/m³")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Forward — compute gravity anomaly on a regular surface grid
# ──────────────────────────────────────────────────────────────────────────────
region = (-10000, 10000, -10000, 10000)   # meters (20 km × 20 km)
shape = (80, 80)
observation_height = 0.0  # meters (at surface)

# verde.grid_coordinates returns (easting, northing, ...) as 2D arrays
coordinates = vd.grid_coordinates(
    region, shape=shape, extra_coords=observation_height
)

print(f"[INFO] Observation grid: {shape[0]}×{shape[1]} points, region={region}")
print(f"[INFO] Observation height: {observation_height} m")

# Compute gravity using the analytical prism formula
# field="g_z" gives the vertical component in m/s² (or mGal depending on version)
gravity_true = hm.prism_gravity(
    coordinates=coordinates,
    prisms=prisms,
    density=densities,
    field="g_z",
)

# Convert to mGal if needed (harmonica returns SI = m/s²; 1 mGal = 1e-5 m/s²)
# Check magnitude to determine units
gravity_max = np.max(np.abs(gravity_true))
print(f"[INFO] True gravity range: [{gravity_true.min():.6e}, {gravity_true.max():.6e}]")

# If values are very small (SI units), convert to mGal for readability
if gravity_max < 1e-2:
    gravity_true_mgal = gravity_true * 1e5  # convert m/s² → mGal
    unit_label = "mGal"
    noise_level = 0.5  # mGal
    print(f"[INFO] Converted to mGal. Range: [{gravity_true_mgal.min():.4f}, {gravity_true_mgal.max():.4f}] mGal")
else:
    gravity_true_mgal = gravity_true
    unit_label = "mGal" if gravity_max < 100 else "m/s²"
    noise_level = 0.5
    print(f"[INFO] Values in native units. Range: [{gravity_true_mgal.min():.4f}, {gravity_true_mgal.max():.4f}] {unit_label}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Add noise to simulate observations
# ──────────────────────────────────────────────────────────────────────────────
noise = np.random.normal(0, noise_level, gravity_true_mgal.shape)
gravity_noisy = gravity_true_mgal + noise
print(f"[INFO] Added Gaussian noise: σ = {noise_level} {unit_label}")
print(f"[INFO] Noisy gravity range: [{gravity_noisy.min():.4f}, {gravity_noisy.max():.4f}] {unit_label}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Inverse — fit EquivalentSources model
# ──────────────────────────────────────────────────────────────────────────────
print("[INFO] Fitting EquivalentSources model...")

# Flatten the coordinates for fitting (verde gives 2D arrays)
easting_flat = coordinates[0].ravel()
northing_flat = coordinates[1].ravel()
height_flat = np.full_like(easting_flat, observation_height)
data_flat = gravity_noisy.ravel()

fit_coords = (easting_flat, northing_flat, height_flat)

# EquivalentSources: depth controls source placement depth below data points
# damping controls regularization (Tikhonov)
eqs = hm.EquivalentSources(
    depth=5000,      # sources placed 5 km below observation points
    damping=1e-3,    # regularization parameter
)
eqs.fit(fit_coords, data_flat)
print("[INFO] EquivalentSources model fitted successfully")

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Predict / reconstruct gravity field
# ──────────────────────────────────────────────────────────────────────────────
predicted_flat = eqs.predict(fit_coords)
gravity_reconstructed = predicted_flat.reshape(shape)

print(f"[INFO] Reconstructed gravity range: [{gravity_reconstructed.min():.4f}, {gravity_reconstructed.max():.4f}] {unit_label}")

# Also predict on upward-continued surface (to demonstrate interpolation capability)
upward_height = 2000.0  # 2 km above surface
coords_up = vd.grid_coordinates(region, shape=shape, extra_coords=upward_height)
easting_up = coords_up[0].ravel()
northing_up = coords_up[1].ravel()
height_up = np.full_like(easting_up, upward_height)
predicted_upward = eqs.predict((easting_up, northing_up, height_up)).reshape(shape)
print(f"[INFO] Upward continued ({upward_height}m) range: [{predicted_upward.min():.4f}, {predicted_upward.max():.4f}] {unit_label}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────
# Use the true gravity field (in mGal) as ground truth
gt = gravity_true_mgal
recon = gravity_reconstructed
residual = gt - recon

# RMSE
rmse = np.sqrt(np.mean(residual**2))

# Correlation Coefficient
cc = np.corrcoef(gt.ravel(), recon.ravel())[0, 1]

# Normalize both to [0, 1] for PSNR/SSIM computation
gt_min, gt_max = gt.min(), gt.max()
data_range = gt_max - gt_min

if data_range > 0:
    gt_norm = (gt - gt_min) / data_range
    recon_norm = (recon - gt_min) / data_range
    recon_norm = np.clip(recon_norm, 0, 1)
else:
    gt_norm = np.zeros_like(gt)
    recon_norm = np.zeros_like(recon)

# PSNR
psnr_val = psnr_func(gt_norm, recon_norm, data_range=1.0)

# SSIM
ssim_val = ssim_func(gt_norm, recon_norm, data_range=1.0)

print(f"\n{'='*60}")
print(f"  EVALUATION METRICS")
print(f"{'='*60}")
print(f"  PSNR  = {psnr_val:.2f} dB")
print(f"  SSIM  = {ssim_val:.4f}")
print(f"  CC    = {cc:.6f}")
print(f"  RMSE  = {rmse:.4f} {unit_label}")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Save outputs
# ──────────────────────────────────────────────────────────────────────────────
# gt_output.npy  = true gravity field (2D array)
# recon_output.npy = reconstructed gravity field (2D array)
np.save(os.path.join(SANDBOX, "gt_output.npy"), gt)
np.save(os.path.join(SANDBOX, "recon_output.npy"), recon)

# Also save to assets
np.save(os.path.join(ASSETS, "gt_output.npy"), gt)
np.save(os.path.join(ASSETS, "recon_output.npy"), recon)

# Save metrics
metrics = {
    "psnr_db": float(psnr_val),
    "ssim": float(ssim_val),
    "cc": float(cc),
    "rmse_mgal": float(rmse),
    "noise_level_mgal": float(noise_level),
    "n_prisms": len(prisms),
    "grid_shape": list(shape),
    "region_m": list(region),
}
import json
with open(os.path.join(SANDBOX, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(ASSETS, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] Saved gt_output.npy, recon_output.npy, metrics.json")

# ──────────────────────────────────────────────────────────────────────────────
# Step 8: 4-panel visualization
# ──────────────────────────────────────────────────────────────────────────────
easting_km = coordinates[0] / 1000.0
northing_km = coordinates[1] / 1000.0
extent = [easting_km.min(), easting_km.max(), northing_km.min(), northing_km.max()]

# Shared color limits for first 3 panels
vmin = min(gt.min(), gravity_noisy.min(), recon.min())
vmax = max(gt.max(), gravity_noisy.max(), recon.max())

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: True gravity anomaly
ax = axes[0, 0]
im1 = ax.imshow(gt, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("(a) True Gravity Anomaly", fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
plt.colorbar(im1, ax=ax, label=f"Gravity ({unit_label})", shrink=0.85)

# Panel 2: Noisy observations
ax = axes[0, 1]
im2 = ax.imshow(gravity_noisy, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title(f"(b) Noisy Observations (σ={noise_level} {unit_label})", fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
plt.colorbar(im2, ax=ax, label=f"Gravity ({unit_label})", shrink=0.85)

# Panel 3: Reconstructed (Equivalent Sources)
ax = axes[1, 0]
im3 = ax.imshow(recon, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("(c) Equivalent Source Reconstruction", fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
plt.colorbar(im3, ax=ax, label=f"Gravity ({unit_label})", shrink=0.85)

# Panel 4: Residual
ax = axes[1, 1]
res_abs_max = max(abs(residual.min()), abs(residual.max()))
im4 = ax.imshow(residual, extent=extent, origin="lower", cmap="RdBu_r",
                vmin=-res_abs_max, vmax=res_abs_max)
ax.set_title("(d) Residual (True − Reconstructed)", fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
plt.colorbar(im4, ax=ax, label=f"Residual ({unit_label})", shrink=0.85)

fig.suptitle(
    f"Gravity Field Inversion via Equivalent Sources\n"
    f"PSNR={psnr_val:.1f} dB | SSIM={ssim_val:.4f} | CC={cc:.4f} | RMSE={rmse:.3f} {unit_label}",
    fontsize=14, fontweight="bold", y=1.02
)

plt.tight_layout()
vis_path_sandbox = os.path.join(SANDBOX, "vis_result.png")
vis_path_assets = os.path.join(ASSETS, "vis_result.png")
fig.savefig(vis_path_sandbox, dpi=150, bbox_inches="tight")
fig.savefig(vis_path_assets, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved visualization: {vis_path_sandbox}")
print(f"[INFO] Saved visualization: {vis_path_assets}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TASK 163: harmonica_gravity — COMPLETE")
print(f"{'='*60}")
print(f"  Forward model: {len(prisms)} rectangular prisms")
print(f"  Grid: {shape[0]}×{shape[1]} @ {observation_height}m elevation")
print(f"  Inverse: EquivalentSources (depth=5000m, damping=1e-3)")
print(f"  PSNR  = {psnr_val:.2f} dB")
print(f"  SSIM  = {ssim_val:.4f}")
print(f"  CC    = {cc:.6f}")
print(f"  RMSE  = {rmse:.4f} {unit_label}")
print(f"{'='*60}")
