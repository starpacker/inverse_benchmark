#!/usr/bin/env python
"""
Raman Spectral Unmixing with Baseline Correction using rampy.

Inverse problem: Given mixed Raman spectra with baseline drift and noise,
decompose them into pure component spectra and mixing proportions, plus
perform baseline correction.

Forward model:  mixed_spectrum = sum(w_i * component_i) + baseline + noise
Inverse solver: (1) Baseline correction via rampy.baseline (ALS)
                (2) NMF for spectral unmixing to recover components & weights
"""

import matplotlib
matplotlib.use('Agg')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
import rampy as rp
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ─── reproducibility ───
np.random.seed(42)

# ─── output directory ───
os.makedirs('results', exist_ok=True)

# ============================================================
# 1. SYNTHESIZE DATA
# ============================================================

# Wavenumber axis
n_points = 500
wavenumber = np.linspace(200, 1200, n_points)

def gaussian_peak(x, center, width, height):
    """Single Gaussian peak."""
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)

def lorentzian_peak(x, center, width, height):
    """Single Lorentzian peak."""
    return height / (1 + ((x - center) / width) ** 2)

# --- Pure component spectra ---
# Component 1: Mineral A – peaks at 400, 600 cm⁻¹
comp1 = (gaussian_peak(wavenumber, 400, 20, 1.0) +
         lorentzian_peak(wavenumber, 600, 15, 0.7))

# Component 2: Mineral B – peaks at 500, 800 cm⁻¹
comp2 = (gaussian_peak(wavenumber, 500, 25, 0.9) +
         gaussian_peak(wavenumber, 800, 18, 1.1))

# Component 3: Mineral C – peaks at 350, 700, 900 cm⁻¹
comp3 = (lorentzian_peak(wavenumber, 350, 22, 0.8) +
         gaussian_peak(wavenumber, 700, 20, 1.0) +
         lorentzian_peak(wavenumber, 900, 16, 0.6))

# Normalize each component to unit max
comp1 /= comp1.max()
comp2 /= comp2.max()
comp3 /= comp3.max()

pure_components = np.vstack([comp1, comp2, comp3])  # (3, 500)
n_components = 3

# --- Generate mixed spectra ---
n_mixtures = 15

# Random mixing proportions that sum to 1
raw_weights = np.random.dirichlet(alpha=[2, 2, 2], size=n_mixtures)  # (15, 3)
true_weights = raw_weights.copy()

# Mixed spectra (clean, no baseline, no noise)
clean_spectra = true_weights @ pure_components  # (15, 500)

# --- Add polynomial baseline drift ---
x_norm = (wavenumber - wavenumber.mean()) / (wavenumber.max() - wavenumber.min())

baselines = np.zeros((n_mixtures, n_points))
for i in range(n_mixtures):
    # Random 3rd-order polynomial coefficients
    c0 = np.random.uniform(0.05, 0.15)
    c1 = np.random.uniform(-0.1, 0.1)
    c2 = np.random.uniform(0.05, 0.2)
    c3 = np.random.uniform(-0.05, 0.05)
    baselines[i] = c0 + c1 * x_norm + c2 * x_norm**2 + c3 * x_norm**3

# --- Add Gaussian noise (SNR ~ 25) ---
signal_power = np.mean(clean_spectra ** 2, axis=1, keepdims=True)
snr = 25.0
noise_std = np.sqrt(signal_power / snr)
noise = noise_std * np.random.randn(n_mixtures, n_points)

# Final observed spectra
observed_spectra = clean_spectra + baselines + noise

print(f"Synthesized {n_mixtures} mixed spectra, {n_points} wavenumber points")
print(f"True weights shape: {true_weights.shape}")
print(f"Observed spectra shape: {observed_spectra.shape}")

# ============================================================
# 2. INVERSE SOLVER
# ============================================================

# --- Step 1: Baseline correction using rampy ---
corrected_spectra = np.zeros_like(observed_spectra)
fitted_baselines = np.zeros_like(observed_spectra)

for i in range(n_mixtures):
    y_corr, bl = rp.baseline(
        wavenumber, observed_spectra[i],
        method='als', lam=1e7, p=0.001, niter=100
    )
    corrected_spectra[i] = y_corr.flatten()
    fitted_baselines[i] = bl.flatten()

# Clip negative values after baseline removal (physical constraint: intensities ≥ 0)
corrected_spectra = np.clip(corrected_spectra, 0, None)

print("Baseline correction completed.")

# --- Step 2: NMF for spectral unmixing ---
# NMF: V ≈ W * H, where V = (n_samples, n_features)
# W = mixing proportions (n_samples, n_components)
# H = component spectra (n_components, n_features)

# Try multiple NMF initializations and pick the best
best_err = np.inf
best_W = None
best_H = None
for seed in range(20):
    nmf_model = NMF(
        n_components=n_components,
        init='nndsvda',
        max_iter=10000,
        random_state=seed,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H=0.0,
        tol=1e-8,
    )
    W_try = nmf_model.fit_transform(corrected_spectra)
    H_try = nmf_model.components_
    err = nmf_model.reconstruction_err_
    if err < best_err:
        best_err = err
        best_W = W_try
        best_H = H_try
        best_seed = seed

W_nmf = best_W  # (n_mixtures, 3)
H_nmf = best_H  # (3, n_points)

print(f"NMF reconstruction error: {best_err:.6f} (seed={best_seed})")

# --- Normalize NMF results ---
# Normalize H rows to unit max, adjust W accordingly
scale_factors = H_nmf.max(axis=1, keepdims=True)
H_norm = H_nmf / scale_factors            # each component spectrum max = 1
W_norm = W_nmf * scale_factors.T           # compensate in weights

# Normalize W rows to sum to 1 (mixing proportions)
W_sum = W_norm.sum(axis=1, keepdims=True)
W_final = W_norm / W_sum

# ============================================================
# 3. MATCH RECOVERED COMPONENTS TO TRUE COMPONENTS
# ============================================================
# Use correlation-based optimal assignment (Hungarian algorithm)

corr_matrix = np.zeros((n_components, n_components))
for i in range(n_components):
    for j in range(n_components):
        corr_matrix[i, j] = np.corrcoef(pure_components[i], H_norm[j])[0, 1]

# Maximize correlation → minimize negative correlation
cost_matrix = -corr_matrix
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Permute recovered components and weights
H_matched = H_norm[col_ind]
W_matched = W_final[:, col_ind]

print(f"Optimal permutation: true_idx -> recovered_idx = {list(zip(row_ind, col_ind))}")

# ============================================================
# 4. EVALUATION METRICS
# ============================================================

metrics = {}

# --- (a) Baseline-corrected PSNR ---
# Compare corrected spectra vs clean spectra (ground truth without baseline/noise)
mse_baseline = np.mean((corrected_spectra - clean_spectra) ** 2)
max_val = clean_spectra.max()
psnr_baseline = 10 * np.log10(max_val ** 2 / mse_baseline)
metrics['baseline_corrected_PSNR_dB'] = float(round(psnr_baseline, 2))

# Correlation between corrected and clean
cc_baseline_list = []
for i in range(n_mixtures):
    cc = np.corrcoef(corrected_spectra[i], clean_spectra[i])[0, 1]
    cc_baseline_list.append(cc)
metrics['baseline_corrected_mean_CC'] = float(round(np.mean(cc_baseline_list), 4))

# --- (b) Component spectrum correlation ---
comp_cc_list = []
for i in range(n_components):
    cc = np.corrcoef(pure_components[i], H_matched[i])[0, 1]
    comp_cc_list.append(cc)
    metrics[f'component_{i+1}_CC'] = float(round(cc, 4))
metrics['mean_component_CC'] = float(round(np.mean(comp_cc_list), 4))

# --- (c) Mixing proportion relative error ---
re_per_sample = np.abs(W_matched - true_weights) / (np.abs(true_weights) + 1e-10)
mean_re = np.mean(re_per_sample)
metrics['mixing_proportion_mean_RE'] = float(round(mean_re, 4))

# Mixing proportion correlation (per component)
for i in range(n_components):
    cc_w = np.corrcoef(true_weights[:, i], W_matched[:, i])[0, 1]
    metrics[f'weight_component_{i+1}_CC'] = float(round(cc_w, 4))

# Overall PSNR (as a summary metric)
metrics['PSNR'] = metrics['baseline_corrected_PSNR_dB']

print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Save metrics
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ============================================================
# 5. SAVE GROUND TRUTH AND RECONSTRUCTION
# ============================================================

# Ground truth: dict with all true data
gt_data = {
    'wavenumber': wavenumber,
    'pure_components': pure_components,
    'true_weights': true_weights,
    'clean_spectra': clean_spectra,
    'baselines': baselines,
    'observed_spectra': observed_spectra,
}
np.save('results/ground_truth.npy', gt_data, allow_pickle=True)

# Reconstruction: dict with all recovered data
recon_data = {
    'corrected_spectra': corrected_spectra,
    'recovered_components': H_matched,
    'recovered_weights': W_matched,
    'fitted_baselines': fitted_baselines,
    'permutation': col_ind,
}
np.save('results/reconstruction.npy', recon_data, allow_pickle=True)

# ============================================================
# 6. VISUALIZATION – 6 subplots
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- (a) Example mixed spectrum: observed vs corrected vs clean ---
ax = axes[0, 0]
idx = 0  # first mixture
ax.plot(wavenumber, observed_spectra[idx], 'b-', alpha=0.6, label='Observed (with baseline+noise)')
ax.plot(wavenumber, corrected_spectra[idx], 'r-', linewidth=1.5, label='Baseline corrected')
ax.plot(wavenumber, clean_spectra[idx], 'k--', linewidth=1.5, label='Ground truth (clean)')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity')
ax.set_title('(a) Baseline Correction Example')
ax.legend(fontsize=8)

# --- (b) Baseline correction detail ---
ax = axes[0, 1]
ax.plot(wavenumber, observed_spectra[idx], 'b-', alpha=0.5, label='Observed')
ax.plot(wavenumber, fitted_baselines[idx], 'g-', linewidth=2, label='Fitted baseline')
ax.plot(wavenumber, baselines[idx], 'm--', linewidth=2, label='True baseline')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity')
ax.set_title('(b) True vs Fitted Baseline')
ax.legend(fontsize=8)

# --- (c) True vs recovered component spectra ---
ax = axes[0, 2]
colors = ['#e41a1c', '#377eb8', '#4daf4a']
labels = ['Mineral A', 'Mineral B', 'Mineral C']
for i in range(n_components):
    ax.plot(wavenumber, pure_components[i], '-', color=colors[i],
            linewidth=2, label=f'True {labels[i]}')
    ax.plot(wavenumber, H_matched[i], '--', color=colors[i],
            linewidth=1.5, alpha=0.8, label=f'Recovered (CC={comp_cc_list[i]:.3f})')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Normalized Intensity')
ax.set_title('(c) True vs Recovered Components')
ax.legend(fontsize=7, ncol=2)

# --- (d) Mixing proportions: true vs recovered (scatter) ---
ax = axes[1, 0]
for i in range(n_components):
    ax.scatter(true_weights[:, i], W_matched[:, i], color=colors[i],
               s=40, label=f'{labels[i]}', alpha=0.8, edgecolors='k', linewidth=0.3)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Ideal')
ax.set_xlabel('True Mixing Proportion')
ax.set_ylabel('Recovered Mixing Proportion')
ax.set_title('(d) Mixing Proportions: True vs Recovered')
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect('equal')

# --- (e) Residual after baseline correction ---
ax = axes[1, 1]
residuals = corrected_spectra - clean_spectra
for i in range(min(5, n_mixtures)):
    ax.plot(wavenumber, residuals[i], alpha=0.5, linewidth=0.8, label=f'Mix {i+1}')
ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Residual')
ax.set_title(f'(e) Baseline Correction Residuals (PSNR={psnr_baseline:.1f} dB)')
ax.legend(fontsize=7)

# --- (f) Per-mixture weight recovery bar chart ---
ax = axes[1, 2]
mix_indices = np.arange(n_mixtures)
bar_width = 0.12
for i in range(n_components):
    ax.bar(mix_indices - bar_width + i * bar_width, true_weights[:, i],
           bar_width, color=colors[i], alpha=0.5, label=f'True {labels[i]}')
    ax.bar(mix_indices + i * bar_width, W_matched[:, i],
           bar_width, color=colors[i], edgecolor='k', linewidth=0.5,
           label=f'Rec {labels[i]}')
ax.set_xlabel('Mixture Index')
ax.set_ylabel('Weight')
ax.set_title('(f) Weight Recovery per Mixture')
# Simplify legend
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles[:6], lbls[:6], fontsize=6, ncol=2)
ax.set_xticks(mix_indices)

plt.suptitle('Raman Spectral Unmixing & Baseline Correction (rampy + NMF)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/reconstruction_result.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nVisualization saved to results/reconstruction_result.png")
print("Metrics saved to results/metrics.json")
print("Ground truth saved to results/ground_truth.npy")
print("Reconstruction saved to results/reconstruction.npy")
print("\n=== TASK COMPLETE ===")
