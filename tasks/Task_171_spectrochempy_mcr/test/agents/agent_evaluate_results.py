import matplotlib

matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt

import json

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def match_components(S_true, S_recovered):
    """
    Match recovered components to true components using correlation.
    Returns permutation indices and correlation matrix.
    """
    n_comp = S_true.shape[0]
    corr_matrix = np.zeros((n_comp, n_comp))

    for i in range(n_comp):
        for j in range(n_comp):
            s_true_norm = S_true[i] / (np.linalg.norm(S_true[i]) + 1e-12)
            s_rec_norm = S_recovered[j] / (np.linalg.norm(S_recovered[j]) + 1e-12)
            corr_matrix[i, j] = np.abs(np.dot(s_true_norm, s_rec_norm))

    # Greedy matching
    perm = []
    used = set()
    for i in range(n_comp):
        best_j = -1
        best_corr = -1
        for j in range(n_comp):
            if j not in used and corr_matrix[i, j] > best_corr:
                best_corr = corr_matrix[i, j]
                best_j = j
        perm.append(best_j)
        used.add(best_j)

    return perm, corr_matrix

def compute_spectral_cc(S_true, S_recovered, perm):
    """Compute correlation coefficients between true and recovered spectra."""
    ccs = []
    for i, j in enumerate(perm):
        cc = np.corrcoef(S_true[i], S_recovered[j])[0, 1]
        ccs.append(abs(cc))
    return np.array(ccs)

def compute_concentration_re(C_true, C_recovered, perm):
    """Compute relative error of recovered concentrations."""
    res = []
    for i, j in enumerate(perm):
        c_true = C_true[:, i]
        c_rec = C_recovered[:, j]
        # Find optimal scaling factor
        scale = np.dot(c_true, c_rec) / (np.dot(c_rec, c_rec) + 1e-12)
        c_rec_scaled = c_rec * scale
        re = np.linalg.norm(c_true - c_rec_scaled) / (np.linalg.norm(c_true) + 1e-12)
        res.append(re)
    return np.array(res)

def compute_psnr(D_true, D_reconstructed):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((D_true - D_reconstructed) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(D_true))
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr

def compute_lack_of_fit(D, D_reconstructed):
    """Compute lack-of-fit percentage."""
    residual = D - D_reconstructed
    lof = np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100
    return lof

def plot_results(wavelengths, S_true, S_recovered, C_true, C_recovered,
                 D_clean, D_reconstructed, perm, save_path):
    """Create 4-subplot visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    comp_names = ['Component 1', 'Component 2', 'Component 3']

    # Scale recovered components to match true for visualization
    S_rec_scaled = np.zeros_like(S_recovered)
    C_rec_scaled = np.zeros_like(C_recovered)
    for i, j in enumerate(perm):
        s_scale = np.max(S_true[i]) / (np.max(S_recovered[j]) + 1e-12)
        S_rec_scaled[i] = S_recovered[j] * s_scale

        c_scale = np.dot(C_true[:, i], C_recovered[:, j]) / (np.dot(C_recovered[:, j], C_recovered[:, j]) + 1e-12)
        C_rec_scaled[:, i] = C_recovered[:, j] * c_scale

    # (a) True vs recovered spectra
    ax = axes[0, 0]
    for i in range(S_true.shape[0]):
        ax.plot(wavelengths, S_true[i], color=colors[i], linewidth=2,
                label=f'{comp_names[i]} (true)')
        ax.plot(wavelengths, S_rec_scaled[i], color=colors[i], linewidth=1.5,
                linestyle='--', label=f'{comp_names[i]} (recovered)')
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title('(a) True vs Recovered Component Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) True vs recovered concentration profiles
    ax = axes[0, 1]
    samples = np.arange(C_true.shape[0])
    for i in range(C_true.shape[1]):
        ax.plot(samples, C_true[:, i], color=colors[i], linewidth=2,
                label=f'{comp_names[i]} (true)')
        ax.plot(samples, C_rec_scaled[:, i], color=colors[i], linewidth=1.5,
                linestyle='--', label=f'{comp_names[i]} (recovered)')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Concentration', fontsize=11)
    ax.set_title('(b) True vs Recovered Concentration Profiles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (c) Reconstructed vs original D (first 5 samples)
    ax = axes[1, 0]
    n_show = min(5, D_clean.shape[0])
    sample_colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    for i in range(n_show):
        ax.plot(wavelengths, D_clean[i], color=sample_colors[i], linewidth=1.5,
                label=f'Sample {i+1} (true)')
        ax.plot(wavelengths, D_reconstructed[i], color=sample_colors[i],
                linewidth=1, linestyle='--', alpha=0.8)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title('(c) Original vs Reconstructed Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (d) Residual matrix heatmap
    ax = axes[1, 1]
    residual = D_clean - D_reconstructed
    im = ax.imshow(residual, aspect='auto', cmap='RdBu_r',
                   extent=[wavelengths[0], wavelengths[-1], D_clean.shape[0], 0])
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Sample Index', fontsize=11)
    ax.set_title('(d) Residual Matrix (D_true - D_reconstructed)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Residual')

    plt.suptitle('MCR-ALS Spectral Decomposition Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the MCR-ALS decomposition results.
    
    Computes metrics comparing recovered components to ground truth,
    saves results to files, and generates visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data containing:
        - 'wavelengths': wavelength array
        - 'S_true': true pure component spectra
        - 'C_true': true concentration profiles
        - 'D_clean': noise-free data matrix
        - 'snr_db': signal-to-noise ratio
        - 'n_components': number of components
    inversion_result : dict
        Dictionary from run_inversion containing:
        - 'C_recovered': recovered concentration matrix
        - 'S_recovered': recovered spectra matrix
        - 'D_reconstructed': reconstructed data matrix
        - 'method_used': string indicating which method was used
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    wavelengths = data_dict['wavelengths']
    S_true = data_dict['S_true']
    C_true = data_dict['C_true']
    D_clean = data_dict['D_clean']
    snr_db = data_dict['snr_db']
    n_components = data_dict['n_components']
    
    C_recovered = inversion_result['C_recovered']
    S_recovered = inversion_result['S_recovered']
    D_reconstructed = inversion_result['D_reconstructed']
    method_used = inversion_result['method_used']
    
    n_samples = C_true.shape[0]
    n_wavelengths = len(wavelengths)
    
    # Match components
    perm, corr_matrix = match_components(S_true, S_recovered)
    print(f"Component matching (true→recovered): {perm}")
    
    # Spectral correlation coefficients
    spectral_ccs = compute_spectral_cc(S_true, S_recovered, perm)
    mean_cc = np.mean(spectral_ccs)
    print(f"\nSpectral Correlation Coefficients:")
    for i, cc in enumerate(spectral_ccs):
        print(f"  Component {i+1}: {cc:.6f}")
    print(f"  Mean CC: {mean_cc:.6f}")
    
    # Concentration relative error
    conc_res = compute_concentration_re(C_true, C_recovered, perm)
    mean_re = np.mean(conc_res)
    print(f"\nConcentration Relative Errors:")
    for i, re in enumerate(conc_res):
        print(f"  Component {i+1}: {re:.6f}")
    print(f"  Mean RE: {mean_re:.6f}")
    
    # Reconstruction metrics
    psnr = compute_psnr(D_clean, D_reconstructed)
    lof = compute_lack_of_fit(D_clean, D_reconstructed)
    print(f"\nReconstruction PSNR: {psnr:.2f} dB")
    print(f"Lack-of-fit: {lof:.4f}%")
    
    # Save metrics
    metrics = {
        "task": "spectrochempy_mcr",
        "method": method_used,
        "inverse_problem": "MCR-ALS spectral decomposition",
        "n_samples": n_samples,
        "n_components": n_components,
        "n_wavelengths": n_wavelengths,
        "snr_db": snr_db,
        "spectral_cc_per_component": spectral_ccs.tolist(),
        "spectral_cc_mean": float(mean_cc),
        "concentration_re_per_component": conc_res.tolist(),
        "concentration_re_mean": float(mean_re),
        "reconstruction_psnr_db": float(psnr),
        "lack_of_fit_pct": float(lof),
        "component_matching": perm
    }
    
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.json")
    
    # Save ground truth and reconstruction
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), D_clean)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), D_reconstructed)
    print(f"Ground truth saved: {RESULTS_DIR}/ground_truth.npy {D_clean.shape}")
    print(f"Reconstruction saved: {RESULTS_DIR}/reconstruction.npy {D_reconstructed.shape}")
    
    # Visualization
    plot_results(wavelengths, S_true, S_recovered, C_true, C_recovered,
                 D_clean, D_reconstructed, perm,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean Spectral CC: {mean_cc:.6f} (target > 0.95)")
    print(f"Reconstruction PSNR: {psnr:.2f} dB (target > 25)")
    print(f"Mean Concentration RE: {mean_re:.6f}")
    print(f"Lack-of-fit: {lof:.4f}%")
    
    if mean_cc > 0.95 and psnr > 25:
        print("✓ All targets met!")
    else:
        print("✗ Some targets not met.")
    
    return metrics
