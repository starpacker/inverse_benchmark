import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_115_adapt_constitutive"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def evaluate_results(
    strain,
    stress_gt,
    stress_noisy,
    stress_recon,
    params_true,
    params_fit
):
    """
    Compute metrics and generate visualizations.
    
    Metrics computed:
        - PSNR: Peak Signal-to-Noise Ratio on stress curve
        - CC: Correlation coefficient
        - RMSE: Root Mean Square Error
        - Parameter relative errors for E, sigma_y, K, n
    
    Args:
        strain: 1D array of strain values
        stress_gt: 1D array of ground-truth stress
        stress_noisy: 1D array of noisy stress observations
        stress_recon: 1D array of reconstructed stress
        params_true: true parameters [E, sigma_y, K, n]
        params_fit: fitted parameters [E, sigma_y, K, n]
    
    Returns:
        metrics: dictionary containing all computed metrics
    """
    # Compute metrics
    mse = np.mean((stress_gt - stress_recon) ** 2)
    data_range = np.max(stress_gt) - np.min(stress_gt)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))
    
    # Correlation coefficient
    cc = float(np.corrcoef(stress_gt.ravel(), stress_recon.ravel())[0, 1])
    
    # RMSE
    rmse = float(np.sqrt(mse))
    
    # Parameter relative errors
    names = ["E", "sigma_y", "K", "n"]
    param_errors = {}
    for name, pt, pf in zip(names, params_true, params_fit):
        re = abs(pf - pt) / abs(pt) * 100.0
        param_errors[f"{name}_true"] = float(pt)
        param_errors[f"{name}_fitted"] = float(pf)
        param_errors[f"{name}_RE_pct"] = float(re)
    
    metrics = {"PSNR": float(psnr), "CC": float(cc), "RMSE": float(rmse)}
    metrics.update(param_errors)
    
    # Print metrics
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.6f}")
    print(f"  RMSE = {metrics['RMSE']:.4f} MPa")
    for k in ["E", "sigma_y", "K", "n"]:
        print(f"  {k}: true={metrics[f'{k}_true']:.4f}  fit={metrics[f'{k}_fitted']:.4f}  RE={metrics[f'{k}_RE_pct']:.2f}%")
    
    # Save results
    print("[3/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), stress_gt)
        np.save(os.path.join(d, "recon_output.npy"), stress_recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    print("[4/4] Generating visualisation ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (a) GT vs Fitted stress-strain curve
    ax = axes[0]
    ax.plot(strain * 100, stress_gt, "k-", lw=2, label="Ground Truth")
    ax.plot(strain * 100, stress_noisy, ".", color="gray", ms=1, alpha=0.4, label="Noisy Observation")
    ax.plot(strain * 100, stress_recon, "r--", lw=2, label="Fitted Model")
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Stress-Strain Curve  (PSNR={metrics['PSNR']:.1f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Parameter comparison
    ax = axes[1]
    param_names = ["E (MPa)", "σ_y (MPa)", "K (MPa)", "n"]
    keys = ["E", "sigma_y", "K", "n"]
    true_vals = [metrics[f"{k}_true"] for k in keys]
    fit_vals = [metrics[f"{k}_fitted"] for k in keys]
    # Normalise for bar chart
    norm_true = [1.0] * 4
    norm_fit = [f / t if t != 0 else 0 for f, t in zip(fit_vals, true_vals)]
    x = np.arange(4)
    ax.bar(x - 0.18, norm_true, 0.35, label="True", color="steelblue")
    ax.bar(x + 0.18, norm_fit, 0.35, label="Fitted", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, fontsize=9)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Parameter Comparison (normalised)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # (c) Residual
    ax = axes[2]
    residual = stress_gt - stress_recon
    ax.plot(strain * 100, residual, "b-", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Residual Stress (MPa)")
    ax.set_title(f"Residual  (RMSE={metrics['RMSE']:.2f} MPa, CC={metrics['CC']:.4f})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    for p in [
        os.path.join(RESULTS_DIR, "reconstruction_result.png"),
        os.path.join(ASSETS_DIR, "reconstruction_result.png"),
        os.path.join(ASSETS_DIR, "vis_result.png")
    ]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics


# ============================================================
# Forward operator needed to reconstruct ground-truth stress
# ============================================================

def forward_operator(strain, E, sigma_y, K, n):
    eps_y = sigma_y / E
    stress = np.empty_like(strain)
    elastic = strain <= eps_y
    plastic = ~elastic
    stress[elastic] = E * strain[elastic]
    eps_p = strain[plastic] - eps_y
    eps_p = np.maximum(eps_p, 0.0)
    stress[plastic] = sigma_y + K * np.power(eps_p, n)
    return stress


# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/adapt_constitutive_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

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
    print(f"[1] Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"[2] Running agent's run_inversion ...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Check for chained execution
    if len(inner_paths) > 0:
        print(f"[INFO] Chained execution detected with {len(inner_paths)} inner file(s).")
        # For chained execution, agent_output should be callable
        for ip in inner_paths:
            print(f"  Loading inner data: {ip}")
            with open(ip, 'rb') as f:
                inner_data = dill.load(f)
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Inner call failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution
        print("[INFO] Direct execution mode.")
        final_result = agent_output
        std_result = std_output

    # ============================================================
    # Extract params_fit and stress_recon from results
    # ============================================================
    # run_inversion returns (params_fit, stress_recon)
    # final_result = agent output: (params_fit, stress_recon)
    # std_result = standard output: (params_fit, stress_recon)

    # Extract agent results
    if isinstance(final_result, tuple) and len(final_result) == 2:
        agent_params_fit = np.asarray(final_result[0])
        agent_stress_recon = np.asarray(final_result[1])
    else:
        print("ERROR: Unexpected agent output format.")
        sys.exit(1)

    # Extract standard results
    if isinstance(std_result, tuple) and len(std_result) == 2:
        std_params_fit = np.asarray(std_result[0])
        std_stress_recon = np.asarray(std_result[1])
    else:
        print("ERROR: Unexpected standard output format.")
        sys.exit(1)

    # Get strain and stress_obs from inputs
    strain = np.asarray(args[0])
    stress_obs = np.asarray(args[1])

    # We need stress_gt and params_true for evaluate_results.
    # Since the standard output was produced by the reference code on the same noisy data,
    # we use the standard's fitted parameters to reconstruct ground truth stress for comparison.
    # However, the true ground truth params are not directly in the pkl.
    # 
    # Strategy: We use the standard result's reconstruction as "ground truth" for evaluation
    # purposes, since that's our reference. We evaluate both against it.
    # 
    # Actually, a better approach: we use std_params_fit as params_true and std_stress_recon as stress_gt,
    # then evaluate the agent's result against that standard.

    # Use standard result as the ground truth reference
    stress_gt = std_stress_recon
    params_true = std_params_fit

    print("\n=== Evaluating Agent Result ===")
    try:
        metrics_agent = evaluate_results(
            strain=strain,
            stress_gt=stress_gt,
            stress_noisy=stress_obs,
            stress_recon=agent_stress_recon,
            params_true=params_true,
            params_fit=agent_params_fit
        )
    except Exception as e:
        print(f"ERROR: evaluate_results failed for agent: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Evaluating Standard Result (self-comparison) ===")
    try:
        metrics_std = evaluate_results(
            strain=strain,
            stress_gt=stress_gt,
            stress_noisy=stress_obs,
            stress_recon=std_stress_recon,
            params_true=params_true,
            params_fit=std_params_fit
        )
    except Exception as e:
        print(f"ERROR: evaluate_results failed for standard: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Score comparison
    # ============================================================
    score_agent_psnr = metrics_agent['PSNR']
    score_std_psnr = metrics_std['PSNR']
    score_agent_cc = metrics_agent['CC']
    score_std_cc = metrics_std['CC']
    score_agent_rmse = metrics_agent['RMSE']
    score_std_rmse = metrics_std['RMSE']

    print(f"\n{'='*60}")
    print(f"Scores -> Agent PSNR: {score_agent_psnr:.2f} dB, Standard PSNR: {score_std_psnr:.2f} dB")
    print(f"Scores -> Agent CC: {score_agent_cc:.6f}, Standard CC: {score_std_cc:.6f}")
    print(f"Scores -> Agent RMSE: {score_agent_rmse:.4f}, Standard RMSE: {score_std_rmse:.4f}")
    print(f"{'='*60}")

    # Additional direct comparison: compare agent stress_recon vs std stress_recon
    direct_mse = np.mean((agent_stress_recon - std_stress_recon) ** 2)
    direct_rmse = np.sqrt(direct_mse)
    stress_range = np.max(std_stress_recon) - np.min(std_stress_recon)
    relative_error = direct_rmse / (stress_range + 1e-30) * 100.0
    print(f"Direct comparison: RMSE between agent and std stress_recon = {direct_rmse:.4f} MPa")
    print(f"Relative to stress range: {relative_error:.2f}%")

    # Verification criteria:
    # The standard result is perfect (PSNR=inf, CC=1.0, RMSE=0) since it compares against itself.
    # For the agent, we need reasonable reconstruction quality.
    # 
    # We check:
    # 1. Agent PSNR should be high (>30 dB indicates good reconstruction)
    # 2. Agent CC should be close to 1.0 (>0.99)
    # 3. Direct relative error should be small (<5%)

    passed = True
    reasons = []

    # Check CC - should be very close to 1.0
    if score_agent_cc < 0.99:
        reasons.append(f"CC too low: {score_agent_cc:.6f} < 0.99")
        passed = False

    # Check relative error between agent and standard
    if relative_error > 5.0:
        reasons.append(f"Relative error too high: {relative_error:.2f}% > 5%")
        passed = False

    # Check PSNR - for a good fit, should be reasonably high
    if score_agent_psnr < 30.0:
        reasons.append(f"PSNR too low: {score_agent_psnr:.2f} dB < 30 dB")
        passed = False

    if passed:
        print("\n✅ TEST PASSED: Agent's run_inversion performs acceptably.")
        sys.exit(0)
    else:
        print(f"\n❌ TEST FAILED: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)