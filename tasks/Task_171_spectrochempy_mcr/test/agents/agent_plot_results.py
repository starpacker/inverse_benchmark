import matplotlib

matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

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
