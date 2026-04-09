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
# Referee evaluation function (injected verbatim from Reference B)
# ============================================================
def mac_value(phi_a, phi_b):
    """MAC between two mode shape vectors."""
    num = np.dot(phi_a, phi_b) ** 2
    den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b)
    return num / (den + 1e-30)


def evaluate_results(d_gt, d_recon, freqs_gt, freqs_recon, modes_gt, modes_recon,
                     freqs_obs, n_modes, n_elem, L_total, results_dir, assets_dir):
    """
    Evaluate reconstruction quality, compute metrics, save results, and visualize.
    """
    # Compute metrics
    mse = np.mean((d_gt - d_recon) ** 2)
    data_range = max(np.max(d_gt) - np.min(d_gt), 0.01)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))

    if np.std(d_gt) > 1e-10:
        cc = float(np.corrcoef(d_gt, d_recon)[0, 1])
    else:
        cc = 0.0

    rmse = float(np.sqrt(mse))

    freq_rmse = float(np.sqrt(np.mean((freqs_gt - freqs_recon) ** 2)))

    mac_vals = []
    modes_recon_copy = modes_recon.copy()
    for j in range(n_modes):
        if np.dot(modes_gt[:, j], modes_recon_copy[:, j]) < 0:
            modes_recon_copy[:, j] *= -1
        mac_vals.append(mac_value(modes_gt[:, j], modes_recon_copy[:, j]))
    avg_mac = float(np.mean(mac_vals))

    gt_damaged = set(np.where(d_gt > 0.05)[0])
    recon_damaged = set(np.where(d_recon > 0.05)[0])
    if len(gt_damaged) > 0:
        detection_rate = len(gt_damaged & recon_damaged) / len(gt_damaged) * 100
    else:
        detection_rate = 100.0

    metrics = {
        "PSNR": float(psnr),
        "CC": cc,
        "RMSE": rmse,
        "freq_RMSE_Hz": freq_rmse,
        "avg_MAC": avg_mac,
        "damage_detection_pct": detection_rate
    }

    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.6f}")
    print(f"  Freq RMSE = {metrics['freq_RMSE_Hz']:.4f} Hz")
    print(f"  Avg MAC   = {metrics['avg_MAC']:.6f}")
    print(f"  Detection = {metrics['damage_detection_pct']:.0f}%")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), d_gt)
        np.save(os.path.join(d, "recon_output.npy"), d_recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    elem_centers = np.arange(n_elem) + 0.5

    ax = axes[0, 0]
    ax.bar(elem_centers - 0.2, d_gt, 0.4, label="True Damage", color="steelblue", alpha=0.8)
    ax.bar(elem_centers + 0.2, d_recon, 0.4, label="Identified Damage", color="salmon", alpha=0.8)
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Damage Parameter d")
    ax.set_title(f"Damage Identification  (PSNR={metrics['PSNR']:.1f} dB, CC={metrics['CC']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_elem)

    ax = axes[0, 1]
    mode_idx = np.arange(1, n_modes + 1)
    ax.plot(mode_idx, freqs_gt, "bo-", label="GT Frequencies")
    ax.plot(mode_idx, freqs_obs, "g^--", label="Observed (noisy)", alpha=0.7)
    ax.plot(mode_idx, freqs_recon, "rs--", label="Identified Model")
    ax.set_xlabel("Mode Number")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Modal Frequencies  (freq RMSE={metrics['freq_RMSE_Hz']:.2f} Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    n_dof = modes_gt.shape[0]
    x_nodes = np.linspace(0, L_total, n_dof)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for j in range(min(3, n_modes)):
        mg = modes_gt[:, j]
        mr = modes_recon_copy[:, j]
        ax.plot(x_nodes, mg, "-", color=colors[j], lw=2, label=f"Mode {j+1} GT")
        ax.plot(x_nodes, mr, "--", color=colors[j], lw=2, label=f"Mode {j+1} Identified")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Mode Shape Amplitude")
    ax.set_title(f"Mode Shapes  (avg MAC={metrics['avg_MAC']:.4f})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    residual = d_gt - d_recon
    ax.bar(elem_centers, residual, 0.6, color="purple", alpha=0.6)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Damage Error (GT − Identified)")
    ax.set_title(f"Damage Residual  (RMSE={metrics['RMSE']:.4f}, Detection={metrics['damage_detection_pct']:.0f}%)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_elem)

    plt.suptitle("Vibration-Based Damage Identification — FE Model Updating", fontsize=14, y=1.01)
    plt.tight_layout()

    for p in [os.path.join(results_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()

    return metrics


# ============================================================
# Helper to extract objective function value from opt_result
# ============================================================
def get_opt_fun(opt_result):
    """Extract the objective function value from an optimization result,
    handling both scipy OptimizeResult objects and dill-serialized dicts."""
    if opt_result is None:
        return None
    if hasattr(opt_result, 'fun'):
        return float(opt_result.fun)
    if isinstance(opt_result, dict) and 'fun' in opt_result:
        return float(opt_result['fun'])
    return None


def get_opt_success(opt_result):
    """Extract success flag from optimization result."""
    if opt_result is None:
        return None
    if hasattr(opt_result, 'success'):
        return bool(opt_result.success)
    if isinstance(opt_result, dict) and 'success' in opt_result:
        return bool(opt_result['success'])
    return None


# ============================================================
# Unpack output tuple handling both tuple and list
# ============================================================
def unpack_output(output):
    """Unpack the output of run_inversion into (d_recon, freqs_recon, modes_recon, opt_result)."""
    if isinstance(output, (tuple, list)):
        d_recon = np.asarray(output[0])
        freqs_recon = np.asarray(output[1])
        modes_recon = np.asarray(output[2])
        opt_result = output[3] if len(output) > 3 else None
        return d_recon, freqs_recon, modes_recon, opt_result
    else:
        raise ValueError(f"Unexpected output type: {type(output)}")


def main():
    data_paths = ['/data/yjh/pyfemu_vibration_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Classify files
    outer_files = []
    inner_files = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)

    if not outer_files:
        print("ERROR: No outer (primary) data file found.")
        sys.exit(1)

    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # Load outer data
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    func_name = outer_data.get('func_name', 'unknown')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Function name: {func_name}")
    print(f"Number of args: {len(args)}, kwargs keys: {list(kwargs.keys())}")

    if inner_files:
        # Pattern 2: Chained Execution
        print("Detected Pattern 2: Chained Execution")
        print("Running run_inversion to get operator...")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"FATAL ERROR running outer function: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_output_inner = inner_data.get('output', None)

        print("Running operator with inner data...")
        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"FATAL ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)

        std_result = std_output_inner
    else:
        # Pattern 1: Direct Execution
        print("Detected Pattern 1: Direct Execution")
        print("Running run_inversion...")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"FATAL ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)

        std_result = std_output

    # Unpack results
    try:
        agent_d, agent_freqs, agent_modes, agent_opt = unpack_output(agent_result)
    except Exception as e:
        print(f"FATAL ERROR unpacking agent result: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        std_d, std_freqs, std_modes, std_opt = unpack_output(std_result)
    except Exception as e:
        print(f"FATAL ERROR unpacking standard result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs for evaluation
    freqs_obs = np.asarray(args[0])
    modes_obs = np.asarray(args[1])
    params = args[2] if len(args) > 2 else kwargs.get('params', {})

    n_elem = params['n_elem']
    L_total = params['L_total']
    n_modes = params['n_modes']

    # Use std_d as ground truth for evaluation (the standard/reference reconstruction)
    d_gt = std_d

    # ============================================================
    # Evaluate AGENT results
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluating AGENT results against STANDARD ground truth...")
    print("=" * 60)
    try:
        metrics_agent = evaluate_results(
            d_gt=d_gt,
            d_recon=agent_d,
            freqs_gt=std_freqs,
            freqs_recon=agent_freqs,
            modes_gt=std_modes,
            modes_recon=agent_modes,
            freqs_obs=freqs_obs,
            n_modes=n_modes,
            n_elem=n_elem,
            L_total=L_total,
            results_dir=os.path.join(results_dir, "agent"),
            assets_dir=os.path.join(assets_dir, "agent")
        )
    except Exception as e:
        print(f"FATAL ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Evaluate STANDARD results (self-consistency check)
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluating STANDARD results (self-consistency check)...")
    print("=" * 60)
    try:
        metrics_std = evaluate_results(
            d_gt=d_gt,
            d_recon=std_d,
            freqs_gt=std_freqs,
            freqs_recon=std_freqs,
            modes_gt=std_modes,
            modes_recon=std_modes,
            freqs_obs=freqs_obs,
            n_modes=n_modes,
            n_elem=n_elem,
            L_total=L_total,
            results_dir=os.path.join(results_dir, "standard"),
            assets_dir=os.path.join(assets_dir, "standard")
        )
    except Exception as e:
        print(f"FATAL ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Direct comparison between Agent and Standard outputs
    # ============================================================
    print("\n" + "=" * 60)
    print("Direct comparison between Agent and Standard outputs...")
    print("=" * 60)

    damage_rmse = float(np.sqrt(np.mean((agent_d - std_d) ** 2)))
    damage_max_diff = float(np.max(np.abs(agent_d - std_d)))
    freq_rmse_vs_std = float(np.sqrt(np.mean((agent_freqs - std_freqs) ** 2)))

    if np.std(std_d) > 1e-10 and np.std(agent_d) > 1e-10:
        cc_vs_std = float(np.corrcoef(agent_d, std_d)[0, 1])
    elif np.allclose(agent_d, std_d, atol=1e-8):
        cc_vs_std = 1.0
    else:
        cc_vs_std = 0.0

    print(f"  Damage RMSE (agent vs std): {damage_rmse:.6f}")
    print(f"  Damage Max Diff: {damage_max_diff:.6f}")
    print(f"  Frequency RMSE (agent vs std): {freq_rmse_vs_std:.6f} Hz")
    print(f"  CC (agent vs std): {cc_vs_std:.6f}")

    # Extract optimization objective values safely
    agent_fun = get_opt_fun(agent_opt)
    std_fun = get_opt_fun(std_opt)

    if agent_fun is not None and std_fun is not None:
        print(f"  Objective value (agent): {agent_fun:.10e}")
        print(f"  Objective value (std):   {std_fun:.10e}")
    else:
        print(f"  Objective value (agent): {agent_fun}")
        print(f"  Objective value (std):   {std_fun}")

    agent_success = get_opt_success(agent_opt)
    std_success = get_opt_success(std_opt)
    print(f"  Optimization success (agent): {agent_success}")
    print(f"  Optimization success (std):   {std_success}")

    # ============================================================
    # Verification & Decision
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Key metrics (higher is better for PSNR, CC, avg_MAC, detection_pct)
    # Lower is better for RMSE, freq_RMSE_Hz
    all_passed = True
    failure_reasons = []

    # 1. PSNR check - higher is better, allow 10% margin
    agent_psnr = metrics_agent["PSNR"]
    # For self-comparison, std PSNR is infinity (or very high). 
    # Instead, check that agent PSNR is reasonably high.
    print(f"  Agent PSNR: {agent_psnr:.2f} dB")
    if agent_psnr < 20.0:  # A reasonable threshold for good reconstruction
        failure_reasons.append(f"PSNR too low: {agent_psnr:.2f} dB (threshold: 20 dB)")
        all_passed = False

    # 2. CC check
    agent_cc = metrics_agent["CC"]
    print(f"  Agent CC: {agent_cc:.6f}")
    if agent_cc < 0.9:
        failure_reasons.append(f"CC too low: {agent_cc:.6f} (threshold: 0.9)")
        all_passed = False

    # 3. avg_MAC check
    agent_mac = metrics_agent["avg_MAC"]
    print(f"  Agent avg MAC: {agent_mac:.6f}")
    if agent_mac < 0.95:
        failure_reasons.append(f"avg_MAC too low: {agent_mac:.6f} (threshold: 0.95)")
        all_passed = False

    # 4. Detection rate check
    agent_det = metrics_agent["damage_detection_pct"]
    print(f"  Agent Detection: {agent_det:.0f}%")
    if agent_det < 80.0:
        failure_reasons.append(f"Detection rate too low: {agent_det:.0f}% (threshold: 80%)")
        all_passed = False

    # 5. Direct comparison: damage RMSE should be small
    print(f"  Damage RMSE (agent vs std): {damage_rmse:.6f}")
    if damage_rmse > 0.1:
        failure_reasons.append(f"Damage RMSE vs std too high: {damage_rmse:.6f} (threshold: 0.1)")
        all_passed = False

    # 6. Objective function comparison (lower is better)
    if agent_fun is not None and std_fun is not None:
        # Allow agent to be at most 50% worse than standard
        if std_fun > 1e-15:
            ratio = agent_fun / std_fun
            print(f"  Objective ratio (agent/std): {ratio:.4f}")
            if ratio > 1.5:
                failure_reasons.append(f"Objective value ratio too high: {ratio:.4f} (threshold: 1.5)")
                all_passed = False
        else:
            # std_fun is essentially zero
            if agent_fun > 1e-6:
                failure_reasons.append(f"Agent objective too high: {agent_fun:.2e} while std is ~0")
                all_passed = False

    # 7. Frequency comparison
    print(f"  Frequency RMSE (agent vs std): {freq_rmse_vs_std:.6f} Hz")
    if freq_rmse_vs_std > 1.0:
        failure_reasons.append(f"Frequency RMSE vs std too high: {freq_rmse_vs_std:.6f} Hz (threshold: 1.0)")
        all_passed = False

    print()
    if all_passed:
        print("=" * 60)
        print("RESULT: PASS - Agent performance is acceptable.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("RESULT: FAIL - Agent performance degraded.")
        for reason in failure_reasons:
            print(f"  - {reason}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)