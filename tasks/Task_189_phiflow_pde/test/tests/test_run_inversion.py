import sys
import os
import dill
import numpy as np
import traceback
import json

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import target function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_run_inversion import run_inversion


def evaluate_results(data_dict, result_dict, save_dir=None):
    """
    Evaluate and visualize the inversion results.
    """
    gt_u0 = data_dict['gt_u0']
    u_obs = data_dict['u_obs']
    u_T = data_dict['u_T']
    params = data_dict['params']

    recon = result_dict['best_recon']
    loss_history = result_dict['loss_history']
    psnr_history = result_dict['psnr_history']

    # Compute final metrics
    gt_np = gt_u0.detach().cpu().numpy().astype(np.float64)
    recon_np = recon.detach().cpu().numpy().astype(np.float64)
    data_range = max(gt_np.max() - gt_np.min(), 1e-10)

    final_psnr = psnr(gt_np, recon_np, data_range=data_range)
    final_ssim = ssim(gt_np, recon_np, data_range=data_range)

    print("")
    print("=" * 50)
    print("Final Results:")
    print("  PSNR: {:.2f} dB".format(final_psnr))
    print("  SSIM: {:.4f}".format(final_ssim))
    print("=" * 50)

    # Setup save directory
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics
    metrics = {
        "psnr": round(float(final_psnr), 2),
        "ssim": round(float(final_ssim), 4),
        "noise_level": params['noise_level'],
        "n_iters": len(loss_history),
        "grid_size": [params['nx'], params['ny']],
        "alpha": params['alpha'],
        "n_steps": params['n_steps'],
        "method": "differentiable_pde_inversion_pytorch_autograd"
    }

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    # Save arrays
    obs_np = u_obs.cpu().numpy()
    u_T_np = u_T.cpu().numpy()

    np.save(os.path.join(save_dir, 'ground_truth.npy'), gt_np)
    np.save(os.path.join(save_dir, 'reconstruction.npy'), recon_np)
    np.save(os.path.join(save_dir, 'observation.npy'), obs_np)
    print("Saved .npy arrays")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmin = min(gt_np.min(), recon_np.min())
    vmax = max(gt_np.max(), recon_np.max())

    im0 = axes[0, 0].imshow(gt_np.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth u0', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(recon_np.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Recovered u0 (PSNR={:.1f}dB)'.format(final_psnr), fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    diff = np.abs(gt_np - recon_np)
    im2 = axes[0, 2].imshow(diff.T, origin='lower', cmap='viridis')
    axes[0, 2].set_title('|Error| (max={:.4f})'.format(diff.max()), fontsize=14)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(obs_np.T, origin='lower', cmap='hot')
    axes[1, 0].set_title('Noisy Observation u(T)+noise', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(u_T_np.T, origin='lower', cmap='hot')
    axes[1, 1].set_title('Clean u(T) (forward solution)', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].semilogy(loss_history)
    axes[1, 2].set_title('Optimization Loss', fontsize=14)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('MSE Loss')
    axes[1, 2].grid(True, alpha=0.3)

    if psnr_history:
        ax_twin = axes[1, 2].twinx()
        iters, psnrs = zip(*psnr_history)
        ax_twin.plot(iters, psnrs, 'r-o', markersize=3, label='PSNR')
        ax_twin.set_ylabel('PSNR (dB)', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')

    nx, ny = params['nx'], params['ny']
    alpha_val = params['alpha']

    suptitle_str = (
        'Heat Equation Initial Condition Inversion\n'
        'du/dt = alpha*laplacian(u), alpha={}, Grid={}x{}, '
        'PSNR={:.2f}dB, SSIM={:.4f}'.format(alpha_val, nx, ny, final_psnr, final_ssim)
    )
    plt.suptitle(suptitle_str, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved reconstruction_result.png")

    print("\nAll results saved to {}/".format(save_dir))

    return metrics


def main():
    data_paths = ['/data/yjh/phiflow_pde_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print("Loading outer data from: {}".format(outer_path))
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print("Running agent run_inversion...")
    try:
        agent_output = run_inversion(*outer_args, **outer_kwargs)
    except Exception as e:
        print("ERROR running run_inversion: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Chained execution pattern
        print("Chained execution detected. Running inner call...")
        inner_path = inner_paths[0]
        print("Loading inner data from: {}".format(inner_path))
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print("ERROR running inner call: {}".format(e))
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        print("Direct execution pattern.")
        final_result = agent_output
        std_result = std_output

    # Extract data_dict from outer_args for evaluation
    # The first arg to run_inversion is data_dict
    data_dict = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('data_dict', None)

    if data_dict is None:
        print("ERROR: Could not extract data_dict for evaluation.")
        sys.exit(1)

    # Evaluate agent result
    print("\n--- Evaluating Agent Result ---")
    try:
        agent_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_agent')
        agent_metrics = evaluate_results(data_dict, final_result, save_dir=agent_save_dir)
    except Exception as e:
        print("ERROR evaluating agent result: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard result
    print("\n--- Evaluating Standard Result ---")
    try:
        std_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_std')
        std_metrics = evaluate_results(data_dict, std_result, save_dir=std_save_dir)
    except Exception as e:
        print("ERROR evaluating standard result: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    # Compare scores
    score_agent_psnr = agent_metrics['psnr']
    score_std_psnr = std_metrics['psnr']
    score_agent_ssim = agent_metrics['ssim']
    score_std_ssim = std_metrics['ssim']

    print("\n" + "=" * 60)
    print("Scores -> Agent PSNR: {}, Standard PSNR: {}".format(score_agent_psnr, score_std_psnr))
    print("Scores -> Agent SSIM: {}, Standard SSIM: {}".format(score_agent_ssim, score_std_ssim))
    print("=" * 60)

    # PSNR: higher is better. Allow 10% margin.
    psnr_threshold = score_std_psnr * 0.9
    ssim_threshold = score_std_ssim * 0.9

    passed = True
    if score_agent_psnr < psnr_threshold:
        print("FAIL: Agent PSNR ({:.2f}) is below threshold ({:.2f}, 90% of standard {:.2f})".format(
            score_agent_psnr, psnr_threshold, score_std_psnr))
        passed = False
    else:
        print("PASS: Agent PSNR ({:.2f}) meets threshold ({:.2f})".format(score_agent_psnr, psnr_threshold))

    if score_agent_ssim < ssim_threshold:
        print("FAIL: Agent SSIM ({:.4f}) is below threshold ({:.4f}, 90% of standard {:.4f})".format(
            score_agent_ssim, ssim_threshold, score_std_ssim))
        passed = False
    else:
        print("PASS: Agent SSIM ({:.4f}) meets threshold ({:.4f})".format(score_agent_ssim, ssim_threshold))

    if passed:
        print("\nOVERALL: PASS")
        sys.exit(0)
    else:
        print("\nOVERALL: FAIL")
        sys.exit(1)


if __name__ == '__main__':
    main()