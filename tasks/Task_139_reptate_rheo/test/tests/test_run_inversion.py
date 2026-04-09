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

# ── Inject the evaluate_results function verbatim from Reference B ──
def evaluate_results(true_params, fitted_params,
                     G_prime_true, G_double_prime_true,
                     G_prime_fit, G_double_prime_fit,
                     omega, G_prime_obs, G_double_prime_obs,
                     results_dir):
    """
    Evaluate inversion results: compute metrics, save outputs, and generate visualizations.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Compute per-parameter relative errors ──
    param_errors = {}
    for key in ('G0', 'tau_R', 'eta_s'):
        tv = true_params[key]
        fv = fitted_params[key]
        re = abs(tv - fv) / abs(tv)
        param_errors[key] = {'true': float(tv), 'fitted': float(fv), 'rel_error': float(re)}

    mean_re = float(np.mean([v['rel_error'] for v in param_errors.values()]))

    # ── Concatenate G' and G'' for spectral metrics (log scale) ──
    EPS = 1e-30
    log_true = np.log10(np.concatenate([G_prime_true, G_double_prime_true]) + EPS)
    log_fit = np.log10(np.concatenate([G_prime_fit, G_double_prime_fit]) + EPS)

    data_range = float(log_true.max() - log_true.min())
    mse = float(np.mean((log_true - log_fit) ** 2))
    psnr = 10.0 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')

    cc = float(np.corrcoef(log_true, log_fit)[0, 1])

    metrics = {
        'psnr_dB': float(psnr),
        'correlation_coefficient': cc,
        'mean_parameter_relative_error': mean_re,
        'parameters': param_errors,
        'method': 'Rouse_model_differential_evolution_fitting',
    }

    # ── Print metrics ──
    print(f"[EVAL] PSNR = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC   = {metrics['correlation_coefficient']:.6f}")
    print(f"[EVAL] Mean RE = {metrics['mean_parameter_relative_error']:.6f}")
    for k, v in metrics['parameters'].items():
        print(f"       {k:>6s}: true={v['true']:.4e}  fitted={v['fitted']:.4e}  RE={v['rel_error']:.6f}")

    # ── Save metrics ──
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # ── Save arrays ──
    np.save(os.path.join(results_dir, "ground_truth.npy"),
            np.column_stack([G_prime_true, G_double_prime_true]))
    np.save(os.path.join(results_dir, "recon_output.npy"),
            np.column_stack([G_prime_fit, G_double_prime_fit]))
    print(f"[SAVE] ground_truth.npy, recon_output.npy → {results_dir}")

    # ── Visualize ──
    vis_path = os.path.join(results_dir, "reconstruction_result.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Storage modulus G'
    ax = axes[0, 0]
    ax.loglog(omega, G_prime_true, 'b-', lw=2, label="G' (true)")
    ax.loglog(omega, G_prime_obs, 'rx', ms=4, alpha=0.5, label="G' (observed)")
    ax.loglog(omega, G_prime_fit, 'g--', lw=2, label="G' (fitted)")
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel("G' (Pa)", fontsize=11)
    ax.set_title("(a) Storage Modulus G'", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # (b) Loss modulus G''
    ax = axes[0, 1]
    ax.loglog(omega, G_double_prime_true, 'b-', lw=2, label="G'' (true)")
    ax.loglog(omega, G_double_prime_obs, 'rx', ms=4, alpha=0.5, label="G'' (observed)")
    ax.loglog(omega, G_double_prime_fit, 'g--', lw=2, label="G'' (fitted)")
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel("G'' (Pa)", fontsize=11)
    ax.set_title("(b) Loss Modulus G''", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # (c) Parameter comparison (bar chart, log scale)
    ax = axes[1, 0]
    params = metrics['parameters']
    names = list(params.keys())
    true_vals = [params[n]['true'] for n in names]
    fit_vals = [params[n]['fitted'] for n in names]
    x_pos = np.arange(len(names))
    w = 0.35
    ax.bar(x_pos - w / 2, true_vals, w, label='True', color='steelblue', alpha=0.8)
    ax.bar(x_pos + w / 2, fit_vals, w, label='Fitted', color='seagreen', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yscale('log')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(c) Parameter Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # (d) Residuals (% relative error vs true, noise-free)
    ax = axes[1, 1]
    res_p = (G_prime_fit - G_prime_true) / G_prime_true * 100.0
    res_pp = (G_double_prime_fit - G_double_prime_true) / G_double_prime_true * 100.0
    ax.semilogx(omega, res_p, 'b.-', ms=3, label="G' residual")
    ax.semilogx(omega, res_pp, 'r.-', ms=3, label="G'' residual")
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('(d) Fit Residuals vs True (noise-free)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Polymer Rheology Inversion (Rouse Model)  |  "
        f"PSNR = {metrics['psnr_dB']:.2f} dB  |  "
        f"CC = {metrics['correlation_coefficient']:.4f}  |  "
        f"Mean RE = {metrics['mean_parameter_relative_error']:.4f}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS]  Saved → {vis_path}")

    return metrics


# ── Forward operator needed to compute ground truth G' and G'' ──
def forward_operator(omega, G0, tau_R, N_modes, eta_s=0.0):
    omega = np.asarray(omega, dtype=np.float64)
    G_prime = np.zeros_like(omega)
    G_double_prime = np.zeros_like(omega)
    for p in range(1, N_modes + 1):
        tau_p = tau_R / p ** 2
        wt = omega * tau_p
        wt2 = wt * wt
        denom = 1.0 + wt2
        G_prime += G0 * wt2 / denom
        G_double_prime += G0 * wt / denom
    G_double_prime += omega * eta_s
    return G_prime, G_double_prime


def main():
    data_paths = ['/data/yjh/reptate_rheo_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    results_dir_agent = './results_agent'
    results_dir_std = './results_std'

    # ── Separate outer and inner data files ──
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

    # ── Load outer data ──
    print(f"[INFO] Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"[INFO] Outer function: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] args count: {len(args)}, kwargs keys: {list(kwargs.keys())}")

    if len(inner_paths) > 0:
        # ── Pattern 2: Chained Execution ──
        print("[INFO] Detected chained execution pattern (inner data found).")
        # Run outer to get operator
        agent_operator = run_inversion(*args, **kwargs)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"[INFO] Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        # Execute
        agent_result = agent_operator(*inner_args, **inner_kwargs)
    else:
        # ── Pattern 1: Direct Execution ──
        print("[INFO] Detected direct execution pattern.")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] run_inversion failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        std_result = std_output

    # ── Extract results ──
    print("[INFO] Agent result type:", type(agent_result))
    print("[INFO] Std result type:", type(std_result))

    # Both should be dicts with keys: G_prime_fit, G_double_prime_fit, fitted_params
    agent_G_prime_fit = agent_result['G_prime_fit']
    agent_G_double_prime_fit = agent_result['G_double_prime_fit']
    agent_fitted_params = agent_result['fitted_params']

    std_G_prime_fit = std_result['G_prime_fit']
    std_G_double_prime_fit = std_result['G_double_prime_fit']
    std_fitted_params = std_result['fitted_params']

    # ── Extract omega and observed data from args ──
    # run_inversion(omega, G_prime_obs, G_double_prime_obs, N_modes=20)
    omega = np.asarray(args[0], dtype=np.float64)
    G_prime_obs = np.asarray(args[1], dtype=np.float64)
    G_double_prime_obs = np.asarray(args[2], dtype=np.float64)

    # ── We need true params and true (noise-free) G'/G'' for evaluation ──
    # The standard fitted_params from the ground truth run serve as our "true" reference
    # since we don't have separate true_params. We use std_fitted_params as the true params
    # and compute true G'/G'' from them.
    true_params = std_fitted_params
    N_modes_true = true_params.get('N_modes', 20)
    G_prime_true, G_double_prime_true = forward_operator(
        omega, true_params['G0'], true_params['tau_R'], N_modes_true, true_params['eta_s']
    )

    # ── Evaluate Agent ──
    print("\n" + "="*60)
    print("EVALUATING AGENT RESULT")
    print("="*60)
    metrics_agent = evaluate_results(
        true_params=true_params,
        fitted_params=agent_fitted_params,
        G_prime_true=G_prime_true,
        G_double_prime_true=G_double_prime_true,
        G_prime_fit=agent_G_prime_fit,
        G_double_prime_fit=agent_G_double_prime_fit,
        omega=omega,
        G_prime_obs=G_prime_obs,
        G_double_prime_obs=G_double_prime_obs,
        results_dir=results_dir_agent,
    )

    # ── Evaluate Standard ──
    print("\n" + "="*60)
    print("EVALUATING STANDARD RESULT")
    print("="*60)
    metrics_std = evaluate_results(
        true_params=true_params,
        fitted_params=std_fitted_params,
        G_prime_true=G_prime_true,
        G_double_prime_true=G_double_prime_true,
        G_prime_fit=std_G_prime_fit,
        G_double_prime_fit=std_G_double_prime_fit,
        omega=omega,
        G_prime_obs=G_prime_obs,
        G_double_prime_obs=G_double_prime_obs,
        results_dir=results_dir_std,
    )

    # ── Compare Metrics ──
    psnr_agent = metrics_agent['psnr_dB']
    psnr_std = metrics_std['psnr_dB']
    cc_agent = metrics_agent['correlation_coefficient']
    cc_std = metrics_std['correlation_coefficient']
    re_agent = metrics_agent['mean_parameter_relative_error']
    re_std = metrics_std['mean_parameter_relative_error']

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Scores -> Agent PSNR: {psnr_agent:.2f} dB, Standard PSNR: {psnr_std:.2f} dB")
    print(f"Scores -> Agent CC: {cc_agent:.6f}, Standard CC: {cc_std:.6f}")
    print(f"Scores -> Agent Mean RE: {re_agent:.6f}, Standard Mean RE: {re_std:.6f}")

    # ── Determine Success ──
    # PSNR: higher is better. Allow 10% margin.
    # CC: higher is better. Allow small margin.
    # Mean RE: lower is better. Allow 10% margin.
    
    passed = True

    # Check PSNR (higher is better)
    if psnr_std != float('inf'):
        if psnr_agent < psnr_std * 0.9:
            print(f"[FAIL] PSNR degraded: agent={psnr_agent:.2f} < threshold={psnr_std * 0.9:.2f}")
            passed = False
        else:
            print(f"[PASS] PSNR acceptable: agent={psnr_agent:.2f} >= threshold={psnr_std * 0.9:.2f}")
    else:
        # Both should be inf if std is inf
        if psnr_agent == float('inf'):
            print("[PASS] Both PSNR are inf (perfect fit).")
        else:
            print(f"[INFO] Standard PSNR is inf, agent PSNR is {psnr_agent:.2f}")
            # If agent PSNR is still very high, pass
            if psnr_agent > 40:
                print("[PASS] Agent PSNR is sufficiently high.")
            else:
                print("[FAIL] Agent PSNR is too low.")
                passed = False

    # Check CC (higher is better)
    if cc_agent < cc_std * 0.99:
        print(f"[FAIL] CC degraded: agent={cc_agent:.6f} < threshold={cc_std * 0.99:.6f}")
        passed = False
    else:
        print(f"[PASS] CC acceptable: agent={cc_agent:.6f} >= threshold={cc_std * 0.99:.6f}")

    # Check Mean RE (lower is better) - allow agent to be up to 10x worse
    if re_std > 0:
        if re_agent > re_std * 10.0:
            print(f"[FAIL] Mean RE degraded: agent={re_agent:.6f} > threshold={re_std * 10.0:.6f}")
            passed = False
        else:
            print(f"[PASS] Mean RE acceptable: agent={re_agent:.6f} <= threshold={re_std * 10.0:.6f}")
    else:
        if re_agent < 0.01:
            print(f"[PASS] Mean RE near zero: {re_agent:.6f}")
        else:
            print(f"[WARN] Standard RE is 0 but agent RE is {re_agent:.6f}")

    if passed:
        print("\n[RESULT] ALL CHECKS PASSED.")
        sys.exit(0)
    else:
        print("\n[RESULT] SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)