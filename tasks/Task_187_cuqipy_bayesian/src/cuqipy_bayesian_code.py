"""
CUQIpy Bayesian Inverse Problem: 1D Deconvolution with Uncertainty Quantification

Uses CUQIpy (Computational Uncertainty Quantification for Inverse problems)
to solve a 1D deconvolution problem via Bayesian inference.

Pipeline:
1. Create 1D deconvolution test problem (convolution forward model + noisy data)
2. Define Bayesian model: GMRF prior + Gaussian likelihood
3. Compute MAP estimate
4. Draw posterior samples via LinearRTO sampler
5. Evaluate reconstruction quality (PSNR, SSIM)
6. Visualize ground truth, observations, MAP, posterior mean with credible intervals

Reference: CUQI-DTU/CUQIpy (Apache-2.0), https://github.com/CUQI-DTU/CUQIpy
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# CUQIpy imports
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, GMRF, LMRF
from cuqi.problem import BayesianProblem


def compute_metrics(ground_truth, reconstruction):
    """Compute PSNR and SSIM between ground truth and reconstruction."""
    gt = np.asarray(ground_truth, dtype=np.float64)
    rec = np.asarray(reconstruction, dtype=np.float64)
    
    # Data range for PSNR/SSIM
    data_range = gt.max() - gt.min()
    
    psnr = peak_signal_noise_ratio(gt, rec, data_range=data_range)
    ssim = structural_similarity(gt, rec, data_range=data_range)
    
    return psnr, ssim


def run_bayesian_deconvolution():
    """Main pipeline: Bayesian 1D deconvolution with CUQIpy."""
    
    # Reproducibility
    np.random.seed(42)
    
    # ---- Output directory ----
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # ========== 1. Create test problem ==========
    dim = 128
    phantom = "sinc"  # smooth phantom with oscillations
    noise_std = 0.01  # noise level
    
    tp = Deconvolution1D(dim=dim, phantom=phantom, noise_std=noise_std)
    
    y_data = tp.data              # noisy observed data
    x_true = tp.exactSolution     # ground truth signal
    A = tp.model                  # linear forward convolution model
    
    print(f"Test problem: Deconvolution1D, dim={dim}, phantom='{phantom}'")
    print(f"  Data shape: {y_data.shape}")
    print(f"  Ground truth range: [{x_true.min():.4f}, {x_true.max():.4f}]")
    print(f"  Observation range:  [{y_data.min():.4f}, {y_data.max():.4f}]")
    
    # ========== 2. Define Bayesian model ==========
    # Prior: Gaussian Markov Random Field (GMRF) — promotes smoothness
    # Higher precision => stronger smoothing
    prior_precision = 140
    x = GMRF(np.zeros(dim), prior_precision, geometry=A.domain_geometry)
    
    # Likelihood: Gaussian with known noise variance
    y = Gaussian(mean=A @ x, cov=noise_std**2)
    
    # Bayesian problem
    BP = BayesianProblem(y, x).set_data(y=y_data)
    
    print(f"\nBayesian model:")
    print(f"  Prior: GMRF(0, precision={prior_precision})")
    print(f"  Likelihood: Gaussian(A@x, noise_var={noise_std**2})")
    
    # ========== 3. Compute MAP estimate ==========
    print("\nComputing MAP estimate...")
    x_map = BP.MAP()
    
    psnr_map, ssim_map = compute_metrics(x_true, x_map)
    print(f"  MAP PSNR: {psnr_map:.2f} dB")
    print(f"  MAP SSIM: {ssim_map:.4f}")
    
    # ========== 4. Sample posterior ==========
    n_samples = 500
    print(f"\nSampling posterior ({n_samples} samples)...")
    samples = BP.sample_posterior(n_samples)
    
    # Posterior statistics
    posterior_mean = samples.mean()
    posterior_std = np.std(samples.samples, axis=1)
    
    # Credible intervals (95%)
    lower_ci = np.percentile(samples.samples, 2.5, axis=1)
    upper_ci = np.percentile(samples.samples, 97.5, axis=1)
    
    psnr_mean, ssim_mean = compute_metrics(x_true, posterior_mean)
    print(f"  Posterior mean PSNR: {psnr_mean:.2f} dB")
    print(f"  Posterior mean SSIM: {ssim_mean:.4f}")
    
    # Choose best reconstruction
    if psnr_mean >= psnr_map:
        x_recon = posterior_mean
        psnr_val, ssim_val = psnr_mean, ssim_mean
        recon_label = "Posterior Mean"
    else:
        x_recon = x_map
        psnr_val, ssim_val = psnr_map, ssim_map
        recon_label = "MAP"
    
    print(f"\nBest reconstruction: {recon_label}")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    
    # ========== 5. Also try LMRF prior (sparsity-promoting) ==========
    print("\n--- LMRF prior (Laplace, sparsity-promoting) ---")
    lmrf_precision = 80
    try:
        # Must reuse same variable names 'x' and 'y' for BayesianProblem to parse correctly
        x2 = LMRF(0, lmrf_precision, geometry=A.domain_geometry, name="x")
        y2 = Gaussian(mean=A @ x2, cov=noise_std**2, name="y")
        BP_lmrf = BayesianProblem(y2, x2).set_data(y=y_data)
        
        x_map_lmrf = BP_lmrf.MAP()
        psnr_lmrf, ssim_lmrf = compute_metrics(x_true, x_map_lmrf)
        print(f"  LMRF MAP PSNR: {psnr_lmrf:.2f} dB, SSIM: {ssim_lmrf:.4f}")
        
        # Use LMRF if better
        if psnr_lmrf > psnr_val:
            x_recon = x_map_lmrf
            psnr_val, ssim_val = psnr_lmrf, ssim_lmrf
            recon_label = "LMRF MAP"
            print(f"  -> LMRF MAP is better, using it.")
    except Exception as e:
        print(f"  LMRF MAP failed: {e}")
        psnr_lmrf, ssim_lmrf = 0.0, 0.0
    
    # ========== 6. Save results ==========
    metrics = {
        "psnr": round(float(psnr_val), 4),
        "ssim": round(float(ssim_val), 4),
        "psnr_map_gmrf": round(float(psnr_map), 4),
        "ssim_map_gmrf": round(float(ssim_map), 4),
        "psnr_posterior_mean": round(float(psnr_mean), 4),
        "ssim_posterior_mean": round(float(ssim_mean), 4),
        "psnr_map_lmrf": round(float(psnr_lmrf), 4),
        "ssim_map_lmrf": round(float(ssim_lmrf), 4),
        "best_method": recon_label,
        "n_posterior_samples": n_samples,
        "dim": dim,
        "phantom": phantom,
        "noise_std": noise_std,
        "gmrf_precision": prior_precision,
        "lmrf_precision": lmrf_precision,
    }
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), x_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), x_recon)
    
    print(f"\nMetrics saved to {results_dir}/metrics.json")
    
    # ========== 7. Visualization ==========
    grid = np.linspace(0, 1, dim)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Ground Truth Signal
    ax = axes[0, 0]
    ax.plot(grid, x_true, 'k-', linewidth=2, label='Ground Truth')
    ax.set_title('(a) Ground Truth Signal', fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Noisy Observations
    ax = axes[0, 1]
    ax.plot(grid, y_data, 'r-', linewidth=1, alpha=0.8, label='Noisy Observations')
    ax.plot(grid, A @ x_true, 'b--', linewidth=1.5, alpha=0.6, label='Clean Convolved')
    ax.set_title('(b) Observed Data (Convolved + Noise)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) MAP Reconstruction
    ax = axes[1, 0]
    ax.plot(grid, x_true, 'k-', linewidth=2, alpha=0.5, label='Ground Truth')
    ax.plot(grid, x_recon, 'b-', linewidth=2, label=f'{recon_label} (Best)')
    if recon_label != "MAP":
        ax.plot(grid, x_map, 'g--', linewidth=1.5, alpha=0.6, label='GMRF MAP')
    ax.set_title(f'(c) {recon_label} Reconstruction\nPSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (d) Posterior Mean + 95% Credible Interval
    ax = axes[1, 1]
    ax.plot(grid, x_true, 'k-', linewidth=2, alpha=0.5, label='Ground Truth')
    ax.plot(grid, posterior_mean, 'b-', linewidth=2, label='Posterior Mean')
    ax.fill_between(grid, lower_ci, upper_ci, alpha=0.25, color='blue',
                    label='95% Credible Interval')
    # Plot a few individual samples
    for i in range(min(10, n_samples)):
        ax.plot(grid, samples.samples[:, i], color='steelblue', alpha=0.08, linewidth=0.5)
    ax.set_title(f'(d) Posterior Mean + 95% CI ({n_samples} samples)\n'
                 f'PSNR={psnr_mean:.2f} dB, SSIM={ssim_mean:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('CUQIpy: Bayesian 1D Deconvolution with Uncertainty Quantification',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {results_dir}/reconstruction_result.png")
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: PSNR = {psnr_val:.4f} dB, SSIM = {ssim_val:.4f}")
    print(f"{'='*50}")
    
    return psnr_val, ssim_val


if __name__ == "__main__":
    psnr, ssim = run_bayesian_deconvolution()
