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
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_results(data, cl_pseudo, result):
    """
    Evaluate reconstruction quality: compute metrics, save results,
    and generate visualization.

    Parameters
    ----------
    data : dict from load_and_preprocess_data
    cl_pseudo : array from forward_operator
    result : dict from run_inversion

    Returns
    -------
    metrics : dict with PSNR, CC, RMSE, mean relative error
    """
    cl_true = data['cl_true']
    lmax = data['lmax']
    cl_recon = result['cl_recon']
    ell_eff = result['ell_eff']

    # Interpolate true Cl at the effective ell values of each bin
    ell_all = np.arange(len(cl_true))
    cl_true_binned = np.interp(ell_eff, ell_all, cl_true)

    # Keep only ell >= 2 (monopole/dipole undefined)
    valid = ell_eff >= 2
    t = cl_true_binned[valid]
    r = cl_recon[valid]

    # PSNR
    data_range = np.max(t) - np.min(t)
    mse = np.mean((t - r) ** 2)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')

    # Pearson CC
    cc = float(np.corrcoef(t, r)[0, 1])

    # Relative error
    re = float(np.mean(np.abs(t - r) / (np.abs(t) + 1e-30)))

    # RMSE
    rmse = float(np.sqrt(mse))

    metrics = {
        "psnr_dB": float(psnr),
        "correlation_coefficient": cc,
        "rmse": rmse,
        "mean_relative_error": re,
        "method": "NaMaster_pseudo_Cl_deconvolution",
    }

    # Print metrics
    print(f"[EVAL] PSNR  = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC    = {metrics['correlation_coefficient']:.6f}")
    print(f"[EVAL] RMSE  = {metrics['rmse']:.6e}")
    print(f"[EVAL] RE    = {metrics['mean_relative_error']:.4f}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # Visualization: four-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cl_true_at_ell = np.interp(ell_eff, ell_all, cl_true)

    def D(ell_arr, cl_arr):
        return ell_arr * (ell_arr + 1) * cl_arr / (2 * np.pi)

    # (a) True power spectrum
    ax = axes[0, 0]
    ax.plot(ell_all[2:], D(ell_all[2:], cl_true[2:]), 'b-', lw=1.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(a) True Power Spectrum')
    ax.set_xlim([2, lmax])

    # (b) Pseudo-Cl (naive) vs True
    ax = axes[0, 1]
    ax.plot(ell_all[2:], D(ell_all[2:], cl_pseudo[2:]), 'r-', alpha=0.7, lw=1, label='Pseudo-Cℓ')
    ax.plot(ell_all[2:], D(ell_all[2:], cl_true[2:]), 'b--', alpha=0.5, lw=1, label='True')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(b) Pseudo-Cℓ (biased) vs True')
    ax.legend()
    ax.set_xlim([2, lmax])

    # (c) Decoupled Cl (NaMaster) vs True
    ax = axes[1, 0]
    ax.plot(ell_eff, D(ell_eff, cl_recon), 'go-', ms=3, lw=1.5, label='NaMaster')
    ax.plot(ell_eff, D(ell_eff, cl_true_at_ell), 'b--', lw=1, label='True')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(c) Decoupled Cℓ (NaMaster) vs True')
    ax.legend()
    ax.set_xlim([2, lmax])

    # (d) Relative error per bin
    ax = axes[1, 1]
    rel_err = np.abs(cl_recon[valid] - cl_true_at_ell[valid]) / (np.abs(cl_true_at_ell[valid]) + 1e-30)
    ax.semilogy(ell_eff[valid], rel_err, 'k.-', ms=3)
    ax.axhline(y=0.1, color='r', ls='--', alpha=0.5, label='10% error')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('|Cℓ_recon − Cℓ_true| / Cℓ_true')
    ax.set_title('(d) Relative Error per ℓ-bin')
    ax.legend()
    ax.set_xlim([2, lmax])

    fig.suptitle(
        f"NaMaster Pseudo-Cℓ Deconvolution  |  "
        f"PSNR={metrics['psnr_dB']:.2f} dB  |  "
        f"CC={metrics['correlation_coefficient']:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {vis_path}")

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), cl_true)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), cl_recon)
    np.save(os.path.join(RESULTS_DIR, "observed_data.npy"), cl_pseudo)
    np.save(os.path.join(RESULTS_DIR, "ell_effective.npy"), ell_eff)
    print("[SAVE] Arrays saved (ground_truth, recon_output, observed_data, ell_effective)")

    return metrics


# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = ['/data/yjh/namaster_pseudo_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

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

    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"[INFO] Function name: {func_name}")
    print(f"[INFO] Number of args: {len(args)}, kwargs keys: {list(kwargs.keys())}")

    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("[INFO] Detected chained execution pattern (inner data found).")
        # Run outer to get operator
        agent_operator = run_inversion(*args, **kwargs)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"[LOAD] Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        # Execute operator
        agent_result = agent_operator(*inner_args, **inner_kwargs)
        # For evaluation, the data input is the first arg of outer
        data_for_eval = args[0] if len(args) > 0 else kwargs.get('data', None)
    else:
        # Pattern 1: Direct Execution
        print("[INFO] Detected direct execution pattern.")
        print("[RUN] Running run_inversion with loaded inputs...")
        agent_result = run_inversion(*args, **kwargs)
        std_result = std_output
        data_for_eval = args[0] if len(args) > 0 else kwargs.get('data', None)

    # Verify agent_result is a dict with expected keys
    print(f"[INFO] Agent result type: {type(agent_result)}")
    if isinstance(agent_result, dict):
        print(f"[INFO] Agent result keys: {list(agent_result.keys())}")
    print(f"[INFO] Standard result type: {type(std_result)}")
    if isinstance(std_result, dict):
        print(f"[INFO] Standard result keys: {list(std_result.keys())}")

    # We need 'cl_pseudo' for evaluate_results; try to get it from data
    # The data dict may contain 'cl_pseudo' or we can compute it
    cl_pseudo = data_for_eval.get('cl_pseudo', None) if isinstance(data_for_eval, dict) else None
    if cl_pseudo is None:
        # Try to create a dummy cl_pseudo (zeros) just for evaluation to work
        # Actually, cl_pseudo is only used for visualization, not for metrics
        cl_true = data_for_eval.get('cl_true', None) if isinstance(data_for_eval, dict) else None
        if cl_true is not None:
            cl_pseudo = np.zeros_like(cl_true)
            print("[WARN] cl_pseudo not found in data, using zeros (only affects visualization).")
        else:
            print("[ERROR] Cannot find cl_true in data for evaluation.")
            sys.exit(1)

    # Evaluate agent result
    print("\n" + "=" * 60)
    print("[EVAL] Evaluating AGENT result...")
    print("=" * 60)
    metrics_agent = evaluate_results(data_for_eval, cl_pseudo, agent_result)

    # Evaluate standard result
    print("\n" + "=" * 60)
    print("[EVAL] Evaluating STANDARD result...")
    print("=" * 60)
    metrics_std = evaluate_results(data_for_eval, cl_pseudo, std_result)

    # Extract primary metric (PSNR - higher is better)
    score_agent = metrics_agent['psnr_dB']
    score_std = metrics_std['psnr_dB']
    cc_agent = metrics_agent['correlation_coefficient']
    cc_std = metrics_std['correlation_coefficient']

    print("\n" + "=" * 60)
    print(f"Scores -> Agent PSNR: {score_agent:.2f} dB, Standard PSNR: {score_std:.2f} dB")
    print(f"Scores -> Agent CC: {cc_agent:.6f}, Standard CC: {cc_std:.6f}")
    print("=" * 60)

    # Verification: PSNR is higher-is-better, allow 10% margin
    # For PSNR in dB, we check if agent is within 90% of standard
    psnr_ok = True
    cc_ok = True

    if score_std > 0:
        if score_agent < score_std * 0.9:
            psnr_ok = False
            print(f"[FAIL] Agent PSNR ({score_agent:.2f}) is significantly below Standard ({score_std:.2f}).")
    elif score_std <= 0:
        # If standard PSNR is negative or zero, just check agent isn't much worse
        if score_agent < score_std - abs(score_std) * 0.1 - 1.0:
            psnr_ok = False
            print(f"[FAIL] Agent PSNR ({score_agent:.2f}) is significantly below Standard ({score_std:.2f}).")

    if cc_agent < cc_std * 0.9:
        cc_ok = False
        print(f"[FAIL] Agent CC ({cc_agent:.6f}) is significantly below Standard ({cc_std:.6f}).")

    if psnr_ok and cc_ok:
        print("[PASS] Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("[FAIL] Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        traceback.print_exc()
        sys.exit(1)