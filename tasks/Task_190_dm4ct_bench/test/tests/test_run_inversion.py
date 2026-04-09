import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from agent_run_inversion import run_inversion


def evaluate_results(phantom, recon_fbp, recon_diffusion, sino_noisy, results_dir,
                     n_angles_sparse, n_angles_full, noise_level, n_outer_iter):
    def compute_psnr(ref, test, data_range=None):
        if data_range is None:
            data_range = ref.max() - ref.min()
        mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)

    def compute_ssim(ref, test):
        data_range = ref.max() - ref.min()
        if data_range == 0:
            data_range = 1.0
        return ssim(ref, test, data_range=data_range)

    def compute_rmse(ref, test):
        return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))

    metrics_fbp = {
        "psnr": float(compute_psnr(phantom, np.clip(recon_fbp, 0, 1))),
        "ssim": float(compute_ssim(phantom, np.clip(recon_fbp, 0, 1))),
        "rmse": float(compute_rmse(phantom, np.clip(recon_fbp, 0, 1))),
    }
    metrics_diff = {
        "psnr": float(compute_psnr(phantom, np.clip(recon_diffusion, 0, 1))),
        "ssim": float(compute_ssim(phantom, np.clip(recon_diffusion, 0, 1))),
        "rmse": float(compute_rmse(phantom, np.clip(recon_diffusion, 0, 1))),
    }

    print(f"\n[EVAL] FBP Baseline:  PSNR={metrics_fbp['psnr']:.2f}dB, SSIM={metrics_fbp['ssim']:.4f}")
    print(f"[EVAL] Diffusion-CT:  PSNR={metrics_diff['psnr']:.2f}dB, SSIM={metrics_diff['ssim']:.4f}")
    print(f"[EVAL] Improvement:   dPSNR={metrics_diff['psnr']-metrics_fbp['psnr']:+.2f}dB, "
          f"dSSIM={metrics_diff['ssim']-metrics_fbp['ssim']:+.4f}")

    metrics = {
        "psnr": metrics_diff["psnr"],
        "ssim": metrics_diff["ssim"],
        "rmse": metrics_diff["rmse"],
        "fbp_psnr": metrics_fbp["psnr"],
        "fbp_ssim": metrics_fbp["ssim"],
        "n_angles_sparse": n_angles_sparse,
        "n_angles_full": n_angles_full,
        "noise_level": noise_level,
        "n_iterations": n_outer_iter,
        "method": "Diffusion-style iterative refinement (TV prior + data consistency)",
    }
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics -> {metrics_path}")

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(sino_noisy, cmap='gray', aspect='auto')
    axes[1].set_title('Sparse Sinogram\n({} views)'.format(n_angles_sparse), fontsize=12)
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('Detector')

    axes[2].imshow(np.clip(recon_fbp, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('FBP Baseline\nPSNR={:.1f}dB'.format(metrics_fbp["psnr"]), fontsize=12)
    axes[2].axis('off')

    axes[3].imshow(np.clip(recon_diffusion, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Iterative Recon\nPSNR={:.1f}dB'.format(metrics_diff["psnr"]), fontsize=12)
    axes[3].axis('off')

    error = np.abs(phantom - recon_diffusion)
    axes[4].imshow(error, cmap='hot', vmin=0, vmax=0.3)
    axes[4].set_title('|Error|', fontsize=12)
    axes[4].axis('off')

    fig.suptitle(
        "DM4CT Sparse-View CT | Diffusion-Style: PSNR={:.2f}dB, SSIM={:.4f} | FBP: PSNR={:.2f}dB".format(
            metrics_diff['psnr'], metrics_diff['ssim'], metrics_fbp['psnr']),
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization -> {vis_path}")

    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_diffusion)
    np.save(os.path.join(results_dir, "ground_truth.npy"), phantom)

    return metrics_diff


def main():
    data_paths = ['/data/yjh/dm4ct_bench_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    results_dir = '/data/yjh/dm4ct_bench_sandbox_sandbox/results'
    os.makedirs(results_dir, exist_ok=True)

    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"[LOAD] Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"[INFO] Outer function: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("[MODE] Chained Execution detected.")
        agent_operator = run_inversion(*outer_args, **outer_kwargs)

        inner_path = inner_paths[0]
        print(f"[LOAD] Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        agent_result = agent_operator(*inner_args, **inner_kwargs)
    else:
        # Pattern 1: Direct Execution
        print("[MODE] Direct Execution detected.")
        agent_result = run_inversion(*outer_args, **outer_kwargs)
        std_result = std_output

    # Both agent_result and std_result should be tuples: (recon_diffusion, recon_fbp)
    if isinstance(agent_result, tuple):
        agent_recon_diffusion = agent_result[0]
        agent_recon_fbp = agent_result[1]
    else:
        agent_recon_diffusion = agent_result
        agent_recon_fbp = None

    if isinstance(std_result, tuple):
        std_recon_diffusion = std_result[0]
        std_recon_fbp = std_result[1]
    else:
        std_recon_diffusion = std_result
        std_recon_fbp = None

    # We need the phantom (ground truth) and other params for evaluate_results.
    # The phantom is not directly in the pkl inputs. We need to reconstruct evaluation params.
    # From the gen_data_code context, the function signature is:
    # run_inversion(sino_noisy, angles_sparse, image_size, n_outer_iter, n_tv_iter, tv_weight, dc_step_size)
    sino_noisy = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('sino_noisy')
    angles_sparse = outer_args[1] if len(outer_args) > 1 else outer_kwargs.get('angles_sparse')
    image_size = outer_args[2] if len(outer_args) > 2 else outer_kwargs.get('image_size')
    n_outer_iter = outer_args[3] if len(outer_args) > 3 else outer_kwargs.get('n_outer_iter')

    n_angles_sparse = len(angles_sparse) if angles_sparse is not None else 0

    # Since we don't have the phantom directly, we use the standard reconstruction as a proxy
    # to compute relative quality. We compare agent vs standard using PSNR/SSIM between them.
    # But evaluate_results needs a phantom. We'll use std_recon_diffusion as the reference phantom
    # to measure how close the agent result is to the standard result.

    # For a fair comparison, evaluate both against the standard reconstruction as reference
    print("\n=== Evaluating Agent Result (against standard as reference) ===")
    agent_results_dir = os.path.join(results_dir, 'agent')
    os.makedirs(agent_results_dir, exist_ok=True)

    score_agent = evaluate_results(
        phantom=std_recon_diffusion,
        recon_fbp=agent_recon_fbp if agent_recon_fbp is not None else np.zeros_like(agent_recon_diffusion),
        recon_diffusion=agent_recon_diffusion,
        sino_noisy=sino_noisy,
        results_dir=agent_results_dir,
        n_angles_sparse=n_angles_sparse,
        n_angles_full=n_angles_sparse * 6,
        noise_level=0.0,
        n_outer_iter=n_outer_iter if n_outer_iter is not None else 0
    )

    print("\n=== Evaluating Standard Result (self-comparison, should be perfect) ===")
    std_results_dir = os.path.join(results_dir, 'standard')
    os.makedirs(std_results_dir, exist_ok=True)

    score_std = evaluate_results(
        phantom=std_recon_diffusion,
        recon_fbp=std_recon_fbp if std_recon_fbp is not None else np.zeros_like(std_recon_diffusion),
        recon_diffusion=std_recon_diffusion,
        sino_noisy=sino_noisy,
        results_dir=std_results_dir,
        n_angles_sparse=n_angles_sparse,
        n_angles_full=n_angles_sparse * 6,
        noise_level=0.0,
        n_outer_iter=n_outer_iter if n_outer_iter is not None else 0
    )

    agent_psnr = score_agent['psnr']
    agent_ssim = score_agent['ssim']
    std_psnr = score_std['psnr']
    std_ssim = score_std['ssim']

    print(f"\nScores -> Agent: PSNR={agent_psnr:.2f}dB SSIM={agent_ssim:.4f}, Standard: PSNR={std_psnr:.2f}dB SSIM={std_ssim:.4f}")

    # For PSNR: higher is better. The standard self-comparison will be inf (perfect match).
    # So instead, we check that agent SSIM is very high (close to 1.0) meaning agent ~= standard.
    # A reasonable threshold: SSIM > 0.90 means the agent output is very close to standard.
    ssim_threshold = 0.90

    # Also do a direct numpy comparison as sanity check
    direct_diff = np.mean(np.abs(agent_recon_diffusion - std_recon_diffusion))
    print(f"Direct mean absolute difference: {direct_diff:.6f}")

    if agent_ssim >= ssim_threshold:
        print(f"\n[PASS] Agent SSIM={agent_ssim:.4f} >= threshold {ssim_threshold}. Performance is acceptable.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Agent SSIM={agent_ssim:.4f} < threshold {ssim_threshold}. Performance degraded significantly.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Exception during test execution:\n{traceback.format_exc()}")
        sys.exit(1)