import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the evaluate_results function (Reference B - verbatim)
# ============================================================
def evaluate_results(
    T21_gt,
    residual_poly,
    residual_pca,
    observation,
    T_fg_mK,
    noise,
    frequencies,
    params,
    poly_order,
    n_pca_components,
    results_dir
):
    """
    Evaluate the inversion results with metrics and visualizations.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    n_freq = params['n_freq']
    n_angle = params['n_angle']
    freq_min = params['freq_min']
    freq_max = params['freq_max']
    
    def compute_psnr(gt, recovered):
        data_range = np.max(gt) - np.min(gt)
        mse = np.mean((gt - recovered) ** 2)
        if mse == 0 or data_range == 0:
            return float('inf')
        return 10.0 * np.log10(data_range ** 2 / mse)
    
    def compute_cc(gt, recovered):
        g = gt.ravel() - np.mean(gt)
        r = recovered.ravel() - np.mean(recovered)
        d = np.sqrt(np.sum(g**2) * np.sum(r**2))
        return float(np.sum(g * r) / d) if d > 0 else 0.0
    
    def compute_rmse(gt, recovered):
        return float(np.sqrt(np.mean((gt - recovered) ** 2)))
    
    poly_psnr = compute_psnr(T21_gt, residual_poly)
    poly_cc = compute_cc(T21_gt, residual_poly)
    poly_rmse = compute_rmse(T21_gt, residual_poly)
    
    pca_psnr = compute_psnr(T21_gt, residual_pca)
    pca_cc = compute_cc(T21_gt, residual_pca)
    pca_rmse = compute_rmse(T21_gt, residual_pca)
    
    fg_ratio = np.std(T_fg_mK) / np.std(T21_gt)
    
    metrics = {
        'poly_psnr': float(poly_psnr),
        'poly_cc': float(poly_cc),
        'poly_rmse': float(poly_rmse),
        'pca_psnr': float(pca_psnr),
        'pca_cc': float(pca_cc),
        'pca_rmse': float(pca_rmse),
        'signal_rms_mK': float(np.std(T21_gt)),
        'foreground_signal_ratio': float(fg_ratio),
        'noise_rms_mK': float(np.std(noise)),
        'n_freq': n_freq,
        'n_angle': n_angle,
        'freq_range_MHz': [float(freq_min), float(freq_max)],
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Grid:              {n_freq} freq x {n_angle} angle")
    print(f"  Freq range:        {freq_min}-{freq_max} MHz")
    print(f"  Signal RMS:        {np.std(T21_gt):.2f} mK")
    print(f"  FG/Signal ratio:   {fg_ratio:.0f}x")
    print(f"  Noise RMS:         {np.std(noise):.2f} mK")
    print(f"  -- Polynomial (order={poly_order}) --")
    print(f"     PSNR = {poly_psnr:.2f} dB | CC = {poly_cc:.4f} | RMSE = {poly_rmse:.2f} mK")
    print(f"  -- PCA (n_comp={n_pca_components}) --")
    print(f"     PSNR = {pca_psnr:.2f} dB | CC = {pca_cc:.4f} | RMSE = {pca_rmse:.2f} mK")
    print("=" * 70)
    
    return metrics


def main():
    # ============================================================
    # Step 1: Parse data paths and identify execution pattern
    # ============================================================
    data_paths = ['/data/yjh/21cmfast_tomo_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    outer_data_path = None
    inner_data_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(p)
        else:
            outer_data_path = p
    
    if outer_data_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # ============================================================
    # Step 2: Load outer data
    # ============================================================
    print(f"Loading outer data from: {outer_data_path}")
    with open(outer_data_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function name: {func_name}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # ============================================================
    # Step 3: Execute the agent function
    # ============================================================
    print("\nRunning agent run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================
    # Step 4: Determine if chained or direct execution
    # ============================================================
    if len(inner_data_paths) > 0:
        # Chained execution
        print(f"\nChained execution detected. Inner data files: {inner_data_paths}")
        inner_data_path = inner_data_paths[0]
        with open(inner_data_path, 'rb') as f:
            inner_data = dill.load(f)
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running chained function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution
        print("\nDirect execution pattern.")
        final_result = agent_output
        std_result = std_output
    
    # ============================================================
    # Step 5: Validate outputs exist and have correct keys
    # ============================================================
    print("\nValidating output structure...")
    expected_keys = ['residual_poly', 'fg_estimate_poly', 'residual_pca', 'fg_estimate_pca']
    
    for key in expected_keys:
        if key not in final_result:
            print(f"ERROR: Missing key '{key}' in agent output.")
            sys.exit(1)
        if key not in std_result:
            print(f"WARNING: Missing key '{key}' in standard output.")
    
    print("Output structure OK. Keys:", list(final_result.keys()))
    
    # ============================================================
    # Step 6: Evaluate using the referee function
    # We need T21_gt, observation, T_fg_mK, noise, frequencies, params, etc.
    # These are not directly available from the pkl, so we reconstruct
    # what we can and compute metrics by comparing agent vs standard.
    # ============================================================
    
    # Extract the inputs from the outer data args
    # Based on the function signature:
    # run_inversion(observation, frequencies, freq_ref, poly_order, n_pca_components)
    
    # Reconstruct all arguments
    import inspect
    sig = inspect.signature(run_inversion)
    param_names = list(sig.parameters.keys())
    
    all_args = {}
    for i, name in enumerate(param_names):
        if i < len(args):
            all_args[name] = args[i]
        elif name in kwargs:
            all_args[name] = kwargs[name]
    
    observation = all_args.get('observation')
    frequencies = all_args.get('frequencies')
    freq_ref = all_args.get('freq_ref')
    poly_order = all_args.get('poly_order')
    n_pca_components = all_args.get('n_pca_components')
    
    print(f"\nInput shapes:")
    print(f"  observation: {observation.shape}")
    print(f"  frequencies: {frequencies.shape}")
    print(f"  freq_ref: {freq_ref}")
    print(f"  poly_order: {poly_order}")
    print(f"  n_pca_components: {n_pca_components}")
    
    n_freq, n_angle = observation.shape
    freq_min = float(frequencies.min())
    freq_max = float(frequencies.max())
    
    # We don't have the ground truth T21, T_fg_mK, noise separately.
    # We'll use the standard result as a proxy ground truth for comparison.
    # The key idea: evaluate how close agent results are to the standard results.
    
    # ============================================================
    # Step 6a: Direct numerical comparison (element-wise)
    # ============================================================
    print("\n" + "=" * 70)
    print("NUMERICAL COMPARISON: Agent vs Standard")
    print("=" * 70)
    
    all_pass = True
    
    for key in expected_keys:
        agent_arr = final_result[key]
        std_arr = std_result[key]
        
        if agent_arr.shape != std_arr.shape:
            print(f"  {key}: SHAPE MISMATCH agent={agent_arr.shape} vs std={std_arr.shape}")
            all_pass = False
            continue
        
        max_abs_diff = np.max(np.abs(agent_arr - std_arr))
        mean_abs_diff = np.mean(np.abs(agent_arr - std_arr))
        
        # Relative error
        denom = np.max(np.abs(std_arr))
        if denom > 0:
            rel_error = max_abs_diff / denom
        else:
            rel_error = max_abs_diff
        
        # RMSE
        rmse = np.sqrt(np.mean((agent_arr - std_arr) ** 2))
        
        print(f"  {key}:")
        print(f"    Shape: {agent_arr.shape}")
        print(f"    Max abs diff: {max_abs_diff:.6e}")
        print(f"    Mean abs diff: {mean_abs_diff:.6e}")
        print(f"    Relative error: {rel_error:.6e}")
        print(f"    RMSE: {rmse:.6e}")
        
        # Check if the relative error is within tolerance
        if rel_error > 1e-4:
            print(f"    STATUS: WARNING - relative error > 1e-4")
        else:
            print(f"    STATUS: OK")
    
    # ============================================================
    # Step 6b: Use standard result as GT for evaluation metrics
    # We treat the standard residual_poly and residual_pca as "ground truth"
    # and measure how well the agent recovers them.
    # ============================================================
    
    # Compute PSNR, CC, RMSE for each output
    def compute_psnr(gt, recovered):
        data_range = np.max(gt) - np.min(gt)
        mse = np.mean((gt - recovered) ** 2)
        if mse == 0 or data_range == 0:
            return float('inf')
        return 10.0 * np.log10(data_range ** 2 / mse)
    
    def compute_cc(gt, recovered):
        g = gt.ravel() - np.mean(gt)
        r = recovered.ravel() - np.mean(recovered)
        d = np.sqrt(np.sum(g**2) * np.sum(r**2))
        return float(np.sum(g * r) / d) if d > 0 else 0.0
    
    def compute_rmse(gt, recovered):
        return float(np.sqrt(np.mean((gt - recovered) ** 2)))
    
    print("\n" + "=" * 70)
    print("QUALITY METRICS: Agent output vs Standard output")
    print("=" * 70)
    
    metrics_summary = {}
    for key in expected_keys:
        agent_arr = final_result[key]
        std_arr = std_result[key]
        
        psnr = compute_psnr(std_arr, agent_arr)
        cc = compute_cc(std_arr, agent_arr)
        rmse = compute_rmse(std_arr, agent_arr)
        
        metrics_summary[key] = {'psnr': psnr, 'cc': cc, 'rmse': rmse}
        print(f"  {key}:")
        print(f"    PSNR = {psnr:.2f} dB | CC = {cc:.6f} | RMSE = {rmse:.6e}")
    
    # ============================================================
    # Step 6c: Self-consistency check using evaluate_results
    # Use the standard's residual as a pseudo ground truth
    # ============================================================
    
    # We use the standard poly residual as a proxy for T21_gt
    # and noise as zeros (since we don't have the actual noise)
    # This gives us a way to compare the metrics side by side.
    
    T21_proxy = std_result['residual_pca']  # Use standard PCA residual as proxy GT
    T_fg_proxy = std_result['fg_estimate_pca']  # Use standard FG estimate as proxy
    noise_proxy = np.zeros_like(observation)  # We don't have actual noise
    
    params = {
        'n_freq': n_freq,
        'n_angle': n_angle,
        'freq_min': freq_min,
        'freq_max': freq_max,
    }
    
    results_dir_agent = '/tmp/eval_agent'
    results_dir_std = '/tmp/eval_std'
    
    print("\n--- Evaluating Agent Output ---")
    try:
        metrics_agent = evaluate_results(
            T21_gt=T21_proxy,
            residual_poly=final_result['residual_poly'],
            residual_pca=final_result['residual_pca'],
            observation=observation,
            T_fg_mK=T_fg_proxy,
            noise=noise_proxy,
            frequencies=frequencies,
            params=params,
            poly_order=poly_order,
            n_pca_components=n_pca_components,
            results_dir=results_dir_agent
        )
    except Exception as e:
        print(f"ERROR in evaluate_results for agent: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n--- Evaluating Standard Output ---")
    try:
        metrics_std = evaluate_results(
            T21_gt=T21_proxy,
            residual_poly=std_result['residual_poly'],
            residual_pca=std_result['residual_pca'],
            observation=observation,
            T_fg_mK=T_fg_proxy,
            noise=noise_proxy,
            frequencies=frequencies,
            params=params,
            poly_order=poly_order,
            n_pca_components=n_pca_components,
            results_dir=results_dir_std
        )
    except Exception as e:
        print(f"ERROR in evaluate_results for standard: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================
    # Step 7: Compare and determine pass/fail
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    # Primary metrics to compare (higher is better for PSNR and CC)
    score_keys_higher_better = ['poly_psnr', 'pca_psnr', 'poly_cc', 'pca_cc']
    # Lower is better for RMSE
    score_keys_lower_better = ['poly_rmse', 'pca_rmse']
    
    passed = True
    margin = 0.10  # 10% margin
    
    for key in score_keys_higher_better:
        agent_val = metrics_agent[key]
        std_val = metrics_std[key]
        threshold = std_val * (1.0 - margin) if std_val > 0 else std_val - abs(std_val) * margin
        status = "PASS" if agent_val >= threshold else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  {key}: Agent={agent_val:.4f}, Standard={std_val:.4f}, Threshold={threshold:.4f} -> {status}")
    
    for key in score_keys_lower_better:
        agent_val = metrics_agent[key]
        std_val = metrics_std[key]
        threshold = std_val * (1.0 + margin) if std_val > 0 else std_val + abs(std_val) * margin
        status = "PASS" if agent_val <= threshold else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  {key}: Agent={agent_val:.4f}, Standard={std_val:.4f}, Threshold={threshold:.4f} -> {status}")
    
    # Also check direct numerical agreement
    # If outputs are nearly identical, that's the strongest signal
    for key in expected_keys:
        cc_val = metrics_summary[key]['cc']
        if cc_val < 0.999:
            print(f"  WARNING: Correlation for {key} = {cc_val:.6f} (< 0.999)")
            # Only fail if correlation is very low
            if cc_val < 0.90:
                passed = False
                print(f"  FAIL: Correlation too low for {key}")
    
    print("\n" + "=" * 70)
    if passed:
        print("OVERALL RESULT: PASS")
        print("=" * 70)
        sys.exit(0)
    else:
        print("OVERALL RESULT: FAIL")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()