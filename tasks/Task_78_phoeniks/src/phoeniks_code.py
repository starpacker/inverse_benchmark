"""
phoeniks - THz Material Parameter Extraction
=============================================
Task: Extract complex refractive index n(ω) and extinction coefficient κ(ω)
      from THz Time-Domain Spectroscopy (THz-TDS) measurements.
      
      Forward model: THz pulse propagates through a dielectric slab.
        H(ω) = E_sample(ω) / E_reference(ω) is the measured transfer function.
      
      Inverse problem: Given measured H(ω), recover n(ω) and κ(ω) by minimizing
        the error between modeled and measured transfer function at each frequency.

Repo: https://github.com/puls-lab/phoeniks
Paper: Pupeza et al., Optics Express 15(7), 4335 (2007)

Usage:
    /data/yjh/phoeniks_env/bin/python phoeniks_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

# Add repo to path
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

import phoeniks as pk
from scipy.constants import c as c_0

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EXAMPLE_DIR = os.path.join(REPO_DIR, "examples", "01_Basic_Extraction")
SAMPLE_THICKNESS = 1e-3  # 1 mm sample thickness
FREQ_START = 0.3e12      # 0.3 THz
FREQ_STOP = 3.7e12       # 3.7 THz

# ═══════════════════════════════════════════════════════════
# 2. Data Loading
# ═══════════════════════════════════════════════════════════
def load_data():
    """
    Load artificial THz-TDS data from the phoeniks example directory.
    Returns: (data_obj, ground_truth_dict)
      - data_obj: phoeniks Data object with reference and sample time-domain traces
      - ground_truth_dict: dict with 'frequency', 'n', 'k', 'alpha' arrays
    """
    ref_file = os.path.join(EXAMPLE_DIR, "Artifical_Reference.txt")
    sam_file = os.path.join(EXAMPLE_DIR, "Artifical_Sample_1mm.txt")
    gt_file = os.path.join(EXAMPLE_DIR, "Artifical_n_k_alpha.txt")

    ref = np.loadtxt(ref_file)
    sam = np.loadtxt(sam_file)
    gt_data = np.loadtxt(gt_file)

    time = ref[:, 0]
    td_reference = ref[:, 1]
    td_sample = sam[:, 1]

    # Create phoeniks Data object
    data_obj = pk.thz_data.Data(time, td_reference, td_sample)

    # Ground truth: columns are [frequency, n, k, alpha]
    ground_truth = {
        'frequency': gt_data[:, 0],
        'n': gt_data[:, 1],
        'k': gt_data[:, 2],
        'alpha': gt_data[:, 3],
    }

    return data_obj, ground_truth


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator (THz Transfer Function)
# ═══════════════════════════════════════════════════════════
def forward_operator(n, k, frequency, thickness):
    """
    Compute the transfer function H(ω) for a single dielectric layer.
    H(ω) = t12 * t21 * P(ω) / P_air(ω) * FP(ω)
    
    where:
      - t12, t21: Fresnel transmission coefficients
      - P(ω): propagation through material
      - P_air(ω): propagation through air (same thickness)
      - FP(ω): Fabry-Perot etalon factor
      
    Parameters:
      n: refractive index array (per frequency)
      k: extinction coefficient array (per frequency)
      frequency: frequency array (Hz)
      thickness: sample thickness (m)
    Returns:
      H: complex transfer function array
    """
    omega = 2 * np.pi * frequency
    complex_n = n - 1j * k  # complex refractive index
    
    t12 = 2 / (1 + complex_n)
    t21 = 2 * complex_n / (1 + complex_n)
    r22 = (complex_n - 1) / (1 + complex_n)
    rr = r22 * r22
    tt = t12 * t21
    
    propagation = np.exp(-1j * omega * complex_n * thickness / c_0)
    propagation_air = np.exp(-1j * omega * thickness / c_0)
    
    FP = 1 / (1 - rr * (propagation ** 2))
    H = tt * propagation * FP / propagation_air
    
    return H


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver / Reconstruction (phoeniks Extraction)
# ═══════════════════════════════════════════════════════════
def reconstruct(data_obj, thickness, freq_start, freq_stop):
    """
    Extract n(ω) and κ(ω) from THz-TDS data using phoeniks.
    
    Algorithm:
      1. Window time-domain traces and zero-pad for frequency resolution
      2. Compute and unwrap phase of transfer function H(ω)
      3. Get initial estimates of n, k from analytical approximation
      4. Optimize n, k at each frequency by minimizing error between
         measured and modeled H(ω)
    
    Parameters:
      data_obj: phoeniks Data object
      thickness: sample thickness (m)
      freq_start: start frequency for phase unwrapping (Hz)
      freq_stop: stop frequency for phase unwrapping (Hz)
    Returns:
      dict with 'frequency', 'n', 'k', 'alpha' arrays
    """
    # Window traces and zero-pad for better frequency resolution
    data_obj.window_traces(time_start=10e-12, time_end=90e-12)
    data_obj.pad_zeros(new_frequency_resolution=2e9)
    
    # Create extraction object
    extract_obj = pk.extraction.Extraction(data_obj)
    
    # Unwrap phase in the specified frequency range
    extract_obj.unwrap_phase(frequency_start=freq_start, frequency_stop=freq_stop)
    
    # Get initial n, k estimates
    n_init, k_init = extract_obj.get_initial_nk(thickness=thickness)
    
    # Run frequency-by-frequency optimization
    frequency, n_opt, k_opt, alpha_opt = extract_obj.run_optimization(thickness=thickness)
    
    result = {
        'frequency': frequency,
        'n': n_opt,
        'k': k_opt,
        'alpha': alpha_opt,
    }
    
    return result


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_psnr(ref, test, data_range=None):
    """Compute PSNR (dB) for 1D signals."""
    if data_range is None:
        data_range = ref.max() - ref.min()
    mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_rmse(ref, test):
    """Compute RMSE."""
    return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))


def compute_correlation(ref, test):
    """Compute Pearson correlation coefficient."""
    r = np.corrcoef(ref.flatten(), test.flatten())[0, 1]
    return r


def compute_relative_error(ref, test):
    """Compute relative error ||ref - test|| / ||ref||."""
    return np.linalg.norm(ref - test) / np.linalg.norm(ref)


def interpolate_to_common_freq(gt_freq, gt_values, recon_freq, recon_values):
    """
    Interpolate reconstructed values onto ground truth frequency grid
    for fair comparison.
    """
    # Find overlapping frequency range
    f_min = max(gt_freq.min(), recon_freq.min())
    f_max = min(gt_freq.max(), recon_freq.max())
    
    # Create mask for GT frequencies within range
    mask = (gt_freq >= f_min) & (gt_freq <= f_max)
    common_freq = gt_freq[mask]
    gt_common = gt_values[mask]
    
    # Interpolate reconstruction to common frequencies
    recon_common = np.interp(common_freq, recon_freq, recon_values)
    
    return common_freq, gt_common, recon_common


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(gt, recon, metrics, save_path):
    """
    Generate comprehensive visualization for THz parameter extraction.
    Shows: (a) n(ω) comparison, (b) α(ω) comparison, (c) n residual, (d) α residual
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Convert to THz for display
    gt_freq_THz = gt['frequency'] / 1e12
    recon_freq_THz = recon['frequency'] / 1e12
    
    # (a) Refractive index n(ω) comparison
    ax = axes[0, 0]
    ax.plot(gt_freq_THz, gt['n'], 'b.', markersize=2, alpha=0.6, label='Ground Truth')
    ax.plot(recon_freq_THz, recon['n'], 'r-', linewidth=1.0, label='phoeniks Extraction')
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Refractive Index n')
    ax.set_title('Refractive Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Absorption coefficient α(ω) comparison
    ax = axes[0, 1]
    gt_alpha_cm = gt['alpha'] * 0.01  # Convert to cm⁻¹
    recon_alpha_cm = recon['alpha'] * 0.01
    ax.plot(gt_freq_THz, gt_alpha_cm, 'b.', markersize=2, alpha=0.6, label='Ground Truth')
    ax.plot(recon_freq_THz, recon_alpha_cm, 'r-', linewidth=1.0, label='phoeniks Extraction')
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel(r'Absorption $\alpha$ (cm$^{-1}$)')
    ax.set_title('Absorption Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Interpolate for residual computation
    common_freq_n, gt_n_common, recon_n_common = interpolate_to_common_freq(
        gt['frequency'], gt['n'], recon['frequency'], recon['n'])
    common_freq_a, gt_a_common, recon_a_common = interpolate_to_common_freq(
        gt['frequency'], gt['alpha'] * 0.01, recon['frequency'], recon['alpha'] * 0.01)
    
    # (c) Refractive index residual
    ax = axes[1, 0]
    residual_n = recon_n_common - gt_n_common
    ax.plot(common_freq_n / 1e12, residual_n, 'g-', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Δn (Extracted − GT)')
    ax.set_title(f'n Residual (RMSE={metrics.get("rmse_n", 0):.6f})')
    ax.grid(True, alpha=0.3)
    
    # (d) Absorption residual
    ax = axes[1, 1]
    residual_a = recon_a_common - gt_a_common
    ax.plot(common_freq_a / 1e12, residual_a, 'm-', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel(r'Δα (cm$^{-1}$)')
    ax.set_title(f'α Residual (RMSE={metrics.get("rmse_alpha", 0):.6f})')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"THz-TDS Parameter Extraction — PSNR_n={metrics.get('psnr_n', 0):.2f} dB | "
        f"CC_n={metrics.get('cc_n', 0):.6f} | "
        f"CC_α={metrics.get('cc_alpha', 0):.6f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  phoeniks — THz Material Parameter Extraction")
    print("=" * 60)

    # (a) Load data
    data_obj, ground_truth = load_data()
    print(f"[DATA] Time-domain reference: {data_obj.td_reference.shape}")
    print(f"[DATA] Time-domain sample: {data_obj.td_sample.shape}")
    print(f"[DATA] Ground truth n/k available at {len(ground_truth['frequency'])} frequencies")
    print(f"[DATA] GT frequency range: {ground_truth['frequency'][0]/1e12:.3f} - {ground_truth['frequency'][-1]/1e12:.3f} THz")

    # (b) Run reconstruction (inverse problem)
    print("\n[RECON] Running phoeniks extraction...")
    reconstruction = reconstruct(data_obj, SAMPLE_THICKNESS, FREQ_START, FREQ_STOP)
    print(f"[RECON] Extracted n/k at {len(reconstruction['frequency'])} frequencies")
    print(f"[RECON] Frequency range: {reconstruction['frequency'][0]/1e12:.3f} - {reconstruction['frequency'][-1]/1e12:.3f} THz")

    # (c) Evaluate — interpolate to common frequency grid
    print("\n[EVAL] Computing metrics...")
    
    # Refractive index n
    common_freq_n, gt_n, recon_n = interpolate_to_common_freq(
        ground_truth['frequency'], ground_truth['n'],
        reconstruction['frequency'], reconstruction['n']
    )
    
    # Extinction coefficient k / absorption alpha
    common_freq_a, gt_alpha, recon_alpha = interpolate_to_common_freq(
        ground_truth['frequency'], ground_truth['alpha'] * 0.01,  # to cm^-1
        reconstruction['frequency'], reconstruction['alpha'] * 0.01
    )
    
    # Also evaluate k
    common_freq_k, gt_k, recon_k = interpolate_to_common_freq(
        ground_truth['frequency'], ground_truth['k'],
        reconstruction['frequency'], reconstruction['k']
    )

    metrics = {
        # n metrics
        "psnr_n": float(compute_psnr(gt_n, recon_n)),
        "cc_n": float(compute_correlation(gt_n, recon_n)),
        "rmse_n": float(compute_rmse(gt_n, recon_n)),
        "re_n": float(compute_relative_error(gt_n, recon_n)),
        # k metrics
        "psnr_k": float(compute_psnr(gt_k, recon_k)),
        "cc_k": float(compute_correlation(gt_k, recon_k)),
        "rmse_k": float(compute_rmse(gt_k, recon_k)),
        # alpha metrics
        "cc_alpha": float(compute_correlation(gt_alpha, recon_alpha)),
        "rmse_alpha": float(compute_rmse(gt_alpha, recon_alpha)),
        # Overall
        "psnr": float(compute_psnr(gt_n, recon_n)),
        "rmse": float(compute_rmse(gt_n, recon_n)),
    }

    print(f"[EVAL] n — PSNR = {metrics['psnr_n']:.4f} dB")
    print(f"[EVAL] n — CC   = {metrics['cc_n']:.8f}")
    print(f"[EVAL] n — RMSE = {metrics['rmse_n']:.8f}")
    print(f"[EVAL] n — RE   = {metrics['re_n']:.8f}")
    print(f"[EVAL] k — PSNR = {metrics['psnr_k']:.4f} dB")
    print(f"[EVAL] k — CC   = {metrics['cc_k']:.8f}")
    print(f"[EVAL] α — CC   = {metrics['cc_alpha']:.8f}")
    print(f"[EVAL] α — RMSE = {metrics['rmse_alpha']:.8f}")

    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")

    # (e) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(ground_truth, reconstruction, metrics, vis_path)

    # (f) Save arrays — combine frequency + n + k as ground truth and reconstruction
    gt_array = np.column_stack([common_freq_n, gt_n, gt_k[:len(common_freq_n)]])
    recon_array = np.column_stack([common_freq_n, recon_n, recon_k[:len(common_freq_n)]])
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_array)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_array)
    
    # Also save the measurement (transfer function magnitude + phase)
    measurement = np.column_stack([
        reconstruction['frequency'],
        reconstruction['n'],
        reconstruction['k'],
        reconstruction['alpha']
    ])
    np.save(os.path.join(RESULTS_DIR, "input.npy"), measurement)

    print("=" * 60)
    print("  DONE")
    print("=" * 60)
