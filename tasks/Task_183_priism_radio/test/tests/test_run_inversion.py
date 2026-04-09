import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Referee / Evaluation Code (injected verbatim from Reference B)
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from skimage.metrics import structural_similarity as ssim


def compute_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(gt, recon):
    """Structural similarity index."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)


def compute_cc(gt, recon):
    """Pearson correlation coefficient."""
    g = gt.ravel()
    r = recon.ravel()
    g_m = g - g.mean()
    r_m = r - r.mean()
    num = np.sum(g_m * r_m)
    den = np.sqrt(np.sum(g_m ** 2) * np.sum(r_m ** 2))
    if den < 1e-15:
        return 0.0
    return float(num / den)


def compute_dynamic_range(image, source_mask):
    """Ratio of peak signal to rms in background region."""
    bg = image[~source_mask]
    rms = np.sqrt(np.mean(bg ** 2)) if len(bg) > 0 else 1e-15
    if rms < 1e-15:
        rms = 1e-15
    return float(image.max() / rms)


def make_dirty_image_eval(vis, ui, vi, nx, ny):
    """Create the dirty image (adjoint applied to visibilities), normalized."""
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    psf_grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(psf_grid, (vi, ui), 1.0)
    dirty = np.fft.ifft2(grid).real
    psf = np.fft.ifft2(psf_grid).real
    peak_psf = psf.max()
    if peak_psf > 0:
        dirty /= peak_psf
    return dirty


def make_figure(sky_gt, dirty, recon, u, v, save_path):
    """Create 5-panel figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmin_log = max(sky_gt[sky_gt > 0].min() * 0.1, 1e-4) if np.any(sky_gt > 0) else 1e-4
    vmax = sky_gt.max()

    ax = axes[0, 0]
    im = ax.imshow(sky_gt, origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=vmin_log, vmax=vmax))
    ax.set_title('(a) Ground Truth Sky', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[0, 1]
    im = ax.imshow(dirty, origin='lower', cmap='inferno')
    ax.set_title('(b) Dirty Image', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[0, 2]
    im = ax.imshow(recon, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
    ax.set_title('(c) L1+TSV Reconstruction', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[1, 0]
    error = np.abs(sky_gt - recon)
    im = ax.imshow(error, origin='lower', cmap='hot')
    ax.set_title('(d) Error |GT - Recon|', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='|Error|')

    ax = axes[1, 1]
    ax.scatter(u, v, s=0.3, alpha=0.3, color='cyan', edgecolors='none')
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlabel('u (pixels)', fontsize=11)
    ax.set_ylabel('v (pixels)', fontsize=11)
    ax.set_title('(e) (u,v) Coverage', fontsize=13, fontweight='bold')

    ax = axes[1, 2]
    cy, cx = 64, 64
    ax.plot(sky_gt[cy, :], 'b-', linewidth=1.5, label='GT row')
    ax.plot(recon[cy, :], 'r--', linewidth=1.5, label='Recon row')
    ax.plot(sky_gt[:, cx], 'b:', linewidth=1.5, label='GT col')
    ax.plot(recon[:, cx], 'r:', linewidth=1.5, label='Recon col')
    ax.set_xlabel('Pixel', fontsize=11)
    ax.set_ylabel('Flux', fontsize=11)
    ax.set_title('(f) Cross-section (row/col through center)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('Task 183: Radio Interferometric Imaging (L1+TSV Sparse Reconstruction)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Figure saved to " + str(save_path))


def evaluate_results(sky_gt, recon, dirty, u_unique, v_unique, ui,
                     lambda_l1, lambda_tsv, max_iter, results_dir='results'):
    """
    Compute metrics, save results, and create visualizations.
    Returns:
        metrics: dict with evaluation metrics
    """
    print("Step 6: Computing metrics ...")
    os.makedirs(results_dir, exist_ok=True)
    nx, ny = sky_gt.shape[1], sky_gt.shape[0]

    psnr_val = compute_psnr(sky_gt, recon)
    ssim_val = compute_ssim(sky_gt, recon)
    cc_val = compute_cc(sky_gt, recon)

    source_mask = sky_gt > 0.05 * sky_gt.max()
    dr_val = compute_dynamic_range(recon, source_mask)

    dirty_norm = np.maximum(dirty, 0)
    if dirty_norm.max() > 0:
        dirty_norm = dirty_norm / dirty_norm.max() * sky_gt.max()
    psnr_dirty = compute_psnr(sky_gt, dirty_norm)

    print("  PSNR (reconstruction): {:.2f} dB".format(psnr_val))
    print("  PSNR (dirty image):    {:.2f} dB".format(psnr_dirty))
    print("  SSIM: {:.4f}".format(ssim_val))
    print("  CC:   {:.4f}".format(cc_val))
    print("  Dynamic Range: {:.1f}".format(dr_val))

    metrics = {
        'task_id': 183,
        'task_name': 'priism_radio',
        'method': 'ISTA with L1+TSV regularization',
        'PSNR_dB': round(psnr_val, 2),
        'SSIM': round(ssim_val, 4),
        'CC': round(cc_val, 4),
        'dynamic_range': round(dr_val, 1),
        'PSNR_dirty_dB': round(psnr_dirty, 2),
        'n_uv_points': int(len(ui)),
        'image_size': [nx, ny],
        'lambda_l1': lambda_l1,
        'lambda_tsv': lambda_tsv,
        'max_iter': max_iter,
    }

    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  Metrics saved to " + str(metrics_path))

    print("Step 7: Creating visualization ...")
    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    make_figure(sky_gt, dirty, recon, u_unique, v_unique, fig_path)

    print("Step 8: Saving arrays ...")
    np.save(os.path.join(results_dir, 'ground_truth.npy'), sky_gt)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon)
    np.save(os.path.join(results_dir, 'dirty_image.npy'), dirty)
    np.save(os.path.join(results_dir, 'uv_coords.npy'), np.stack([u_unique, v_unique]))

    print("=" * 60)
    print("DONE. All outputs saved to results/")
    print("  PSNR = {:.2f} dB  (target > 20 dB)".format(psnr_val))
    print("  SSIM = {:.4f}".format(ssim_val))
    print("  CC   = {:.4f}".format(cc_val))

    assert psnr_val > 20.0, "PSNR {:.2f} dB < 20 dB target!".format(psnr_val)
    print("PSNR > 20 dB -- PASS")

    return metrics


# ============================================================
# Main Test Logic
# ============================================================
def main():
    data_paths = [
        '/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl'
    ]

    # Separate outer vs inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Also scan the directory for any inner files we might have missed
    data_dir = os.path.dirname(outer_path)
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            full = os.path.join(data_dir, fname)
            if full not in data_paths and 'parent_function' in fname and 'run_inversion' in fname:
                inner_paths.append(full)

    print("Outer data: " + str(outer_path))
    print("Inner data files: " + str(inner_paths))

    # Load outer data
    print("Loading outer data...")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print("Outer args count: " + str(len(args)))
    print("Outer kwargs keys: " + str(list(kwargs.keys())))

    # Fix the random seed for reproducible step_size estimation
    np.random.seed(42)

    if len(inner_paths) > 0:
        # Pattern 2: Chained execution
        print("Pattern 2: Chained execution detected.")
        operator = run_inversion(*args, **kwargs)

        inner_path = inner_paths[0]
        print("Loading inner data from: " + str(inner_path))
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        final_result = operator(*inner_args, **inner_kwargs)
    else:
        # Pattern 1: Direct execution
        print("Pattern 1: Direct execution.")
        agent_output = run_inversion(*args, **kwargs)
        final_result = agent_output
        std_result = std_output

    # Extract image and history from results
    # run_inversion returns (image, history)
    if isinstance(final_result, tuple):
        agent_image = final_result[0]
        agent_history = final_result[1] if len(final_result) > 1 else {}
    else:
        agent_image = final_result
        agent_history = {}

    if isinstance(std_result, tuple):
        std_image = std_result[0]
        std_history = std_result[1] if len(std_result) > 1 else {}
    else:
        std_image = std_result
        std_history = {}

    print("Agent image shape: " + str(agent_image.shape))
    print("Std image shape: " + str(std_image.shape))
    print("Agent image range: [{:.6f}, {:.6f}]".format(agent_image.min(), agent_image.max()))
    print("Std image range: [{:.6f}, {:.6f}]".format(std_image.min(), std_image.max()))

    # Extract parameters needed for evaluate_results
    # Get vis, ui, vi, nx, ny from the original args/kwargs
    if len(args) >= 5:
        vis = args[0]
        ui = args[1]
        vi = args[2]
        nx = args[3]
        ny = args[4]
    else:
        vis = kwargs.get('vis', args[0] if len(args) > 0 else None)
        ui = kwargs.get('ui', args[1] if len(args) > 1 else None)
        vi = kwargs.get('vi', args[2] if len(args) > 2 else None)
        nx = kwargs.get('nx', args[3] if len(args) > 3 else None)
        ny = kwargs.get('ny', args[4] if len(args) > 4 else None)

    lambda_l1 = kwargs.get('lambda_l1', 2e-4)
    lambda_tsv = kwargs.get('lambda_tsv', 1e-3)
    max_iter = kwargs.get('max_iter', 800)

    if len(args) > 5:
        lambda_l1 = args[5]