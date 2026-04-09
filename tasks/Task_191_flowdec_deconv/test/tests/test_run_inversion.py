import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from agent_run_inversion import run_inversion


def evaluate_results(gt_volume, observed, deconvolved, save_dir='results'):
    """
    Evaluate reconstruction quality and visualize results.
    """
    os.makedirs(save_dir, exist_ok=True)

    mid_z = gt_volume.shape[0] // 2

    gt_slice = gt_volume[mid_z]
    recon_slice = deconvolved[mid_z]
    obs_slice = observed[mid_z]

    vmin, vmax = gt_slice.min(), gt_slice.max()
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0

    gt_norm = (gt_slice - vmin) / (vmax - vmin)
    recon_norm = np.clip((recon_slice - vmin) / (vmax - vmin), 0, 1)
    obs_norm = np.clip((obs_slice - vmin) / (vmax - vmin), 0, 1)

    psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)

    psnr_baseline = psnr(gt_norm, obs_norm, data_range=1.0)
    ssim_baseline = ssim(gt_norm, obs_norm, data_range=1.0)

    metrics = {
        'psnr': float(round(psnr_val, 4)),
        'ssim': float(round(ssim_val, 4)),
        'baseline_psnr': float(round(psnr_baseline, 4)),
        'baseline_ssim': float(round(ssim_baseline, 4))
    }

    gt_s = gt_volume[mid_z]
    obs_s = observed[mid_z]
    dec_s = deconvolved[mid_z]
    err_s = np.abs(gt_s - dec_s)

    vmin_vis, vmax_vis = gt_s.min(), gt_s.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(gt_s, cmap='hot', vmin=vmin_vis, vmax=vmax_vis)
    axes[0].set_title('Ground Truth (z-center)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(obs_s, cmap='hot', vmin=vmin_vis, vmax=vmax_vis)
    axes[1].set_title('Blurred + Noisy Input', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(dec_s, cmap='hot', vmin=vmin_vis, vmax=vmax_vis)
    axes[2].set_title(
        'RL Deconvolved\nPSNR={:.2f} dB, SSIM={:.4f}'.format(metrics["psnr"], metrics["ssim"]),
        fontsize=11
    )
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_s, cmap='viridis')
    axes[3].set_title('|GT - Deconvolved| Error', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('Task 191: 3D Richardson-Lucy Deconvolution (Fluorescence Microscopy)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved to {}".format(save_path))

    metrics_out = {
        'task': 'flowdec_deconv',
        'task_number': 191,
        'method': 'Richardson-Lucy deconvolution (scikit-image)',
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'baseline_psnr': metrics['baseline_psnr'],
        'baseline_ssim': metrics['baseline_ssim']
    }
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print("Metrics saved to {}".format(metrics_path))

    np.save(os.path.join(save_dir, 'ground_truth.npy'), gt_volume)
    np.save(os.path.join(save_dir, 'reconstruction.npy'), deconvolved)
    print("Arrays saved to {}/".format(save_dir))

    return metrics


def main():
    data_paths = [
        '/data/yjh/flowdec_deconv_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl'
    ]

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

    print("Outer function: {}".format(outer_data.get('func_name', 'unknown')))
    print("Number of outer args: {}".format(len(outer_args)))
    print("Outer kwargs keys: {}".format(list(outer_kwargs.keys())))

    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("Detected chained execution pattern with {} inner file(s).".format(len(inner_paths)))

        print("Running run_inversion with outer data...")
        try:
            agent_operator = run_inversion(*outer_args, **outer_kwargs)
        except Exception as e:
            print("ERROR running run_inversion (outer): {}".format(e))
            traceback.print_exc()
            sys.exit(1)

        inner_path = inner_paths[0]
        print("Loading inner data from: {}".format(inner_path))
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        print("Running operator with inner data...")
        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print("ERROR running operator (inner): {}".format(e))
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("Detected direct execution pattern.")

        print("Running run_inversion with outer data...")
        try:
            agent_result = run_inversion(*outer_args, **outer_kwargs)
        except Exception as e:
            print("ERROR running run_inversion: {}".format(e))
            traceback.print_exc()
            sys.exit(1)

        std_result = std_output

    # Now we need gt_volume and observed for evaluate_results
    # The outer_args for run_inversion are: (observed, psf, n_iterations)
    # We need gt_volume which is NOT an input to run_inversion
    # We'll use std_result as a proxy ground truth for comparison
    # observed is the first arg
    observed = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('observed', None)

    if observed is None:
        print("ERROR: Could not extract observed volume from inputs.")
        sys.exit(1)

    # Use std_result as ground truth reference for evaluation
    # Evaluate agent result
    print("Evaluating agent result...")
    try:
        agent_metrics = evaluate_results(
            gt_volume=std_result,
            observed=observed,
            deconvolved=agent_result,
            save_dir='results_agent'
        )
    except Exception as e:
        print("ERROR evaluating agent result: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard result (should be near-perfect since gt == std)
    print("Evaluating standard result...")
    try:
        std_metrics = evaluate_results(
            gt_volume=std_result,
            observed=observed,
            deconvolved=std_result,
            save_dir='results_std'
        )
    except Exception as e:
        print("ERROR evaluating standard result: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    agent_psnr = agent_metrics['psnr']
    agent_ssim = agent_metrics['ssim']
    std_psnr = std_metrics['psnr']
    std_ssim = std_metrics['ssim']

    print("=" * 60)
    print("Scores -> Agent PSNR: {}, Standard PSNR: {}".format(agent_psnr, std_psnr))
    print("Scores -> Agent SSIM: {}, Standard SSIM: {}".format(agent_ssim, std_ssim))
    print("=" * 60)

    # Since we're comparing agent output against std output as GT,
    # high PSNR/SSIM means the agent closely matches the standard.
    # A perfect match would give infinite PSNR and SSIM=1.0.
    # We check that PSNR is reasonably high (> 20 dB) and SSIM > 0.8
    # which indicates the agent's output is very close to the standard.
    psnr_threshold = 20.0
    ssim_threshold = 0.8

    passed = True
    if agent_psnr < psnr_threshold:
        print("FAIL: Agent PSNR ({}) is below threshold ({})".format(agent_psnr, psnr_threshold))
        passed = False
    else:
        print("PASS: Agent PSNR ({}) meets threshold ({})".format(agent_psnr, psnr_threshold))

    if agent_ssim < ssim_threshold:
        print("FAIL: Agent SSIM ({}) is below threshold ({})".format(agent_ssim, ssim_threshold))
        passed = False
    else:
        print("PASS: Agent SSIM ({}) meets threshold ({})".format(agent_ssim, ssim_threshold))

    if passed:
        print("Overall: PASSED - Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("Overall: FAILED - Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == '__main__':
    main()