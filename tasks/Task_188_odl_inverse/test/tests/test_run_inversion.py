import sys
import os
import dill
import numpy as np
import traceback
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

warnings.filterwarnings('ignore')

# Import target function
from agent_run_inversion import run_inversion


def forward_operator(x, ray_transform):
    if isinstance(x, np.ndarray):
        x_element = ray_transform.domain.element(x)
    else:
        x_element = x
    y_pred = ray_transform(x_element)
    return y_pred.asarray()


def evaluate_results(ground_truth, reconstructions, data_params, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def compute_metrics(gt_arr, recon_arr, label):
        data_range = gt_arr.max() - gt_arr.min()
        psnr = float(peak_signal_noise_ratio(gt_arr, recon_arr, data_range=data_range))
        ssim = float(structural_similarity(gt_arr, recon_arr, data_range=data_range))
        print(f'  {label:12s}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}')
        return psnr, ssim

    print('\n=== Evaluation ===')

    recon_fbp = reconstructions['fbp']
    recon_cgls = reconstructions['cgls']
    recon_pdhg = reconstructions['pdhg']

    psnr_fbp, ssim_fbp = compute_metrics(ground_truth, recon_fbp, 'FBP')
    psnr_cgls, ssim_cgls = compute_metrics(ground_truth, recon_cgls, 'CGLS')
    psnr_pdhg, ssim_pdhg = compute_metrics(ground_truth, recon_pdhg, 'TV-PDHG')

    np.save(os.path.join(output_dir, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_pdhg)

    algo_params = reconstructions['parameters']
    metrics = {
        'task': 'odl_inverse',
        'method': 'TV-PDHG (Total Variation via Primal-Dual Hybrid Gradient)',
        'PSNR': round(psnr_pdhg, 4),
        'SSIM': round(ssim_pdhg, 4),
        'all_methods': {
            'FBP': {'PSNR': round(psnr_fbp, 4), 'SSIM': round(ssim_fbp, 4)},
            'CGLS': {'PSNR': round(psnr_cgls, 4), 'SSIM': round(ssim_cgls, 4)},
            'TV-PDHG': {'PSNR': round(psnr_pdhg, 4), 'SSIM': round(ssim_pdhg, 4)},
        },
        'parameters': {
            'image_size': data_params['image_size'],
            'num_angles': data_params['num_angles'],
            'noise_level': data_params['noise_level'],
            'tv_lambda': algo_params['tv_lambda'],
            'pdhg_iterations': algo_params['niter_pdhg'],
        }
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(f'\nMetrics saved -> {os.path.join(output_dir, "metrics.json")}')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    gt = ground_truth
    num_angles = data_params['num_angles']
    noise_level = data_params['noise_level']

    ax = axes[0, 0]
    im = ax.imshow(gt, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title('Ground Truth (Shepp-Logan)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    sinogram_display = forward_operator(ground_truth, data_params['ray_transform'])
    im = ax.imshow(sinogram_display, cmap='gray', aspect='auto')
    ax.set_title(f'Sinogram ({num_angles} angles, {noise_level*100:.0f}% noise)', fontsize=12)
    ax.set_xlabel('Detector pixel')
    ax.set_ylabel('Angle index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    im = ax.imshow(recon_fbp, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title(f'FBP  (PSNR={psnr_fbp:.1f} dB, SSIM={ssim_fbp:.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(recon_cgls, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title(f'CGLS  (PSNR={psnr_cgls:.1f} dB, SSIM={ssim_cgls:.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im = ax.imshow(recon_pdhg, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title(f'TV-PDHG  (PSNR={psnr_pdhg:.1f} dB, SSIM={ssim_pdhg:.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    err = np.abs(gt - recon_pdhg)
    im = ax.imshow(err, cmap='hot')
    ax.set_title('|GT - TV-PDHG| Error Map', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Task 188: ODL Inverse - CT Reconstruction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure saved  -> {os.path.join(output_dir, "reconstruction_result.png")}')

    print('\nTask 188 (odl_inverse) completed successfully.')
    print(f'  Primary result (TV-PDHG): PSNR={psnr_pdhg:.2f} dB, SSIM={ssim_pdhg:.4f}')

    return metrics


def main():
    data_paths = ['/data/yjh/odl_inverse_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    output_dir = '/data/yjh/odl_inverse_sandbox_sandbox/run_code/output'
    std_data_dir = '/data/yjh/odl_inverse_sandbox_sandbox/run_code/std_data'

    # Classify files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # Also scan the directory for inner files
    if os.path.isdir(std_data_dir):
        for fname in os.listdir(std_data_dir):
            full = os.path.join(std_data_dir, fname)
            if full == outer_path:
                continue
            if ('parent_function' in fname or 'parent_' in fname) and fname.endswith('.pkl'):
                if full not in inner_paths:
                    inner_paths.append(full)

    print(f'Outer data: {outer_path}')
    print(f'Inner data files: {inner_paths}')

    # Load outer data
    print('\nLoading outer data...')
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f'Function: {func_name}')
    print(f'Args count: {len(args)}')
    print(f'Kwargs keys: {list(kwargs.keys())}')

    # Extract ground truth and data_params from the input data
    # The first arg should be the data dict
    input_data = args[0] if len(args) > 0 else kwargs.get('data', None)

    # We need ground_truth and data_params for evaluate_results
    # ground_truth should be in the input data or we derive it
    ground_truth = None
    data_params = None

    if isinstance(input_data, dict):
        ground_truth = input_data.get('ground_truth', None)
        data_params = input_data.get('data_params', None)

        # If data_params not directly available, try to construct it
        if data_params is None:
            data_params = {}
            if 'image_size' in input_data:
                data_params['image_size'] = input_data['image_size']
            if 'num_angles' in input_data:
                data_params['num_angles'] = input_data['num_angles']
            if 'noise_level' in input_data:
                data_params['noise_level'] = input_data['noise_level']
            if 'ray_transform' in input_data:
                data_params['ray_transform'] = input_data['ray_transform']

    if len(inner_paths) > 0:
        # Chained execution
        print('\n=== Chained Execution Mode ===')
        print('Running outer function (run_inversion)...')
        agent_output = run_inversion(*args, **kwargs)

        # Load inner data
        inner_path = inner_paths[0]
        print(f'\nLoading inner data from: {inner_path}')
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        print('Running inner function (operator)...')
        final_result = agent_output(*inner_args, **inner_kwargs)

    else:
        # Direct execution
        print('\n=== Direct Execution Mode ===')
        print('Running run_inversion...')
        agent_output = run_inversion(*args, **kwargs)
        final_result = agent_output
        std_result = std_output

    # Now evaluate both agent and standard results
    print('\n=== Evaluating Agent Result ===')
    try:
        if ground_truth is not None and data_params is not None:
            # We have all we need for the full evaluate_results
            # Make sure data_params has required keys
            required_keys = ['image_size', 'num_angles', 'noise_level', 'ray_transform']
            has_all = all(k in data_params for k in required_keys)

            if has_all:
                agent_metrics = evaluate_results(ground_truth, final_result, data_params, os.path.join(output_dir, 'agent'))
                print('\n=== Evaluating Standard Result ===')
                std_metrics = evaluate_results(ground_truth, std_result, data_params, os.path.join(output_dir, 'standard'))

                score_agent = agent_metrics['PSNR']
                score_std = std_metrics['PSNR']
            else:
                raise ValueError('Missing required data_params keys, falling back to direct comparison')
        else:
            raise ValueError('No ground_truth or data_params, falling back to direct comparison')

    except Exception as e:
        print(f'Full evaluation not possible ({e}), using direct PSNR comparison...')

        # Compare reconstructions directly using std_result as reference
        if isinstance(final_result, dict) and isinstance(std_result, dict):
            agent_pdhg = final_result.get('pdhg', final_result.get('primary', None))
            std_pdhg = std_result.get('pdhg', std_result.get('primary', None))

            if agent_pdhg is not None and std_pdhg is not None:
                agent_pdhg = np.asarray(agent_pdhg)
                std_pdhg = np.asarray(std_pdhg)

                data_range = std_pdhg.max() - std_pdhg.min()
                if data_range == 0:
                    data_range = 1.0

                psnr_between = float(peak_signal_noise_ratio(std_pdhg, agent_pdhg, data_range=data_range))
                ssim_between = float(structural_similarity(std_pdhg, agent_pdhg, data_range=data_range))

                print(f'Direct comparison (agent vs standard):')
                print(f'  PSNR between outputs: {psnr_between:.2f} dB')
                print(f'  SSIM between outputs: {ssim_between:.4f}')

                # Also compare FBP and CGLS
                for key in ['fbp', 'cgls']:
                    if key in final_result and key in std_result:
                        a = np.asarray(final_result[key])
                        s = np.asarray(std_result[key])
                        dr = s.max() - s.min()
                        if dr == 0:
                            dr = 1.0
                        p = float(peak_signal_noise_ratio(s, a, data_range=dr))
                        ss = float(structural_similarity(s, a, data_range=dr))
                        print(f'  {key.upper()}: PSNR={p:.2f} dB, SSIM={ss:.4f}')

                # For direct comparison, high PSNR means outputs are very similar
                # PSNR > 30 dB is generally very good agreement
                score_agent = psnr_between
                score_std = 50.0  # Reference: perfect match would be infinity
                # Use SSIM as the primary metric instead
                if ssim_between >= 0.90:
                    print(f'\nScores -> Agent SSIM: {ssim_between:.4f} (threshold: 0.90)')
                    print('PASS: Agent output closely matches standard output.')
                    sys.exit(0)
                else:
                    print(f'\nScores -> Agent SSIM: {ssim_between:.4f} (threshold: 0.90)')
                    print('FAIL: Agent output differs significantly from standard.')
                    sys.exit(1)
            else:
                print('Could not extract reconstruction arrays for comparison.')
                sys.exit(1)
        else:
            print('Results are not in expected dict format.')
            sys.exit(1)

    # Final scoring (when full evaluation was possible)
    print(f'\nScores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}')

    # PSNR is higher-is-better. Allow 10% margin.
    threshold = score_std * 0.90
    print(f'Threshold (90% of standard): {threshold:.4f}')

    if score_agent >= threshold:
        print('PASS: Agent performance is within acceptable range.')
        sys.exit(0)
    else:
        print(f'FAIL: Agent PSNR {score_agent:.4f} < threshold {threshold:.4f}')
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {e}')
        traceback.print_exc()
        sys.exit(1)