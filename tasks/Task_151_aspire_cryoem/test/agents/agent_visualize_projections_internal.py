import matplotlib

matplotlib.use('Agg')

import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_projections_internal(clean_images, noisy_images, save_path):
    """Show sample projection images (clean vs noisy)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    n_images = clean_images.shape[0]
    step = max(1, n_images // 4)

    for j in range(4):
        idx = min(j * step, n_images - 1)
        axes[0, j].imshow(clean_images[idx], cmap='gray')
        axes[0, j].set_title(f'Clean #{idx}', fontsize=11)
        axes[0, j].axis('off')

        axes[1, j].imshow(noisy_images[idx], cmap='gray')
        axes[1, j].set_title(f'Noisy #{idx}', fontsize=11)
        axes[1, j].axis('off')

    fig.suptitle('Sample Cryo-EM Projections (Clean vs Noisy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
