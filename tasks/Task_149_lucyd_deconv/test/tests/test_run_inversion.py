import sys
import os
import dill
import numpy as np
import traceback
import json
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Import target function
from agent_run_inversion import run_inversion

# --- Injected Referee (evaluate_results) ---
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(ground_truth, reconstruction, blurred_noisy, params):
    data_range = 1.0
    gt = np.clip(ground_truth, 0, data_range)
    recon = np.clip(reconstruction, 0, data_range)
    degraded = np.clip(blurred_noisy, 0, data_range)

    psnr_val = psnr_metric(gt, recon, data_range=data_range)
    ssim_val = ssim_metric(gt, recon, data_range=data_range)
    rmse_val = np.sqrt(np.mean((gt - recon) ** 2))

    degraded_psnr = psnr_metric(gt, degraded, data_range=data_range)
    degraded_ssim = ssim_metric(gt, degraded, data_range=data_range)

    metrics = {
        'PSNR': float(round(psnr_val, 2)),
        'SSIM': float(round(ssim_val, 4)),
        'RMSE': float(round(rmse_val, 4)),
        'degraded_PSNR': float(round(degraded_psnr, 2)),
        'degraded_SSIM': float(round(degraded_ssim, 4)),
        'PSNR_improvement': float(round(psnr_val - degraded_psnr, 2)),
        'method': 'Richardson-Lucy with TV regularization',
        'n_iterations': 150,
        'tv_weight': 0.002,
        'psf_sigma': params.get('psf_sigma', 2.5),
        'photon_gain': params.get('photon_gain', 500),
    }

    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), reconstruction)

    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# --- Main Test Logic ---
def main():
    data_paths = [
        '/data/yjh/lucyd_deconv_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl'
    ]

    # Separate outer vs inner files
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
    print(f"Loading outer data: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Running run_inversion with {len(outer_args)} args, {len(outer_kwargs)} kwargs...")

    try:
        agent_output = run_inversion(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR running run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Check for chained execution
    if inner_paths:
        # Pattern 2: Chained
        inner_path = inner_paths[0]
        print(f"Loading inner data: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner call: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct
        final_result = agent_output
        std_result = std_output

    # --- Evaluation ---
    # We need ground_truth and blurred_noisy for evaluate_results.
    # observed (blurred_noisy) is the first arg to run_inversion.
    observed = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('observed')
    params = {
        'psf_sigma': 2.5,
        'photon_gain': 500,
    }

    # The ground truth for evaluation: use std_result as ground truth reference
    # Normalize observed to [0,1] for fair degraded metric comparison
    observed_norm = np.clip(observed, 0, None)
    obs_max = observed_norm.max()
    if obs_max > 0:
        observed_norm = observed_norm / (obs_max + 1e-12)

    print("Evaluating agent reconstruction...")
    metrics_agent = evaluate_results(std_result, final_result, observed_norm, params)

    print("Evaluating standard reconstruction (self-check)...")
    metrics_std = evaluate_results(std_result, std_result, observed_norm, params)

    agent_psnr = metrics_agent['PSNR']
    std_psnr = metrics_std['PSNR']
    agent_ssim = metrics_agent['SSIM']
    std_ssim = metrics_std['SSIM']

    print(f"Scores -> Agent PSNR: {agent_psnr}, Standard PSNR: {std_psnr}")
    print(f"Scores -> Agent SSIM: {agent_ssim}, Standard SSIM: {std_ssim}")
    print(f"Agent RMSE: {metrics_agent['RMSE']}")

    # Verification: PSNR higher is better. Allow 10% margin.
    # std_psnr is inf (perfect match with itself), so we check agent quality directly.
    # A reasonable reconstruction should have high PSNR against the reference.
    # We check that RMSE is very small (reconstructions should be nearly identical).
    rmse = metrics_agent['RMSE']
    print(f"RMSE between agent and standard output: {rmse}")

    if rmse > 0.05:
        print(f"FAIL: RMSE {rmse} exceeds tolerance 0.05")
        sys.exit(1)

    if agent_psnr < 20.0:
        print(f"FAIL: Agent PSNR {agent_psnr} is too low (< 20 dB)")
        sys.exit(1)

    print("PASS: Agent reconstruction quality is acceptable.")
    sys.exit(0)


if __name__ == '__main__':
    main()