import sys
import os
import dill
import numpy as np
import traceback
import time
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the agent's function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "logA"]

PARAM_LABELS = [r"$H_0$", r"$\Omega_b h^2$", r"$\Omega_c h^2$",
                r"$n_s$", r"$\ln(10^{10}A_s)$"]

def evaluate_results(data, inversion_results, total_runtime, results_dir):
    """
    Compute metrics and generate visualizations.
    
    Parameters
    ----------
    data : dict
        Output from load_and_preprocess_data
    inversion_results : dict
        Output from run_inversion
    total_runtime : float
        Total runtime in seconds
    results_dir : str
        Directory to save results
    
    Returns
    -------
    dict
        Metrics dictionary
    """
    print("[4/5] Computing metrics ...")
    
    os.makedirs(results_dir, exist_ok=True)
    
    Dl_true = data['Dl_true']
    Dl_obs = data['Dl_obs']
    ells = data['ells']
    lmin = data['lmin']
    lmax = data['lmax']
    
    Dl_recon = inversion_results['Dl_recon']
    pr = inversion_results['parameter_results']
    
    dt = Dl_true[lmin:lmax + 1]
    dr = Dl_recon[lmin:lmax + 1]
    mse = np.mean((dt - dr)**2)
    psnr = 10 * np.log10(np.max(dt)**2 / mse) if mse > 0 else float('inf')
    corr = float(np.corrcoef(dt, dr)[0, 1])
    
    pe = {}
    for p in PARAM_NAMES:
        r = pr[p]
        re = abs(r['median'] - r['true']) / abs(r['true']) * 100
        pe[p] = {
            'true': r['true'],
            'estimated': round(r['median'], 6),
            'relative_error_pct': round(re, 4),
            'std': round(r['std'], 6)
        }
    mre = np.mean([v['relative_error_pct'] for v in pe.values()])
    
    metrics = {
        'psnr_dB': round(float(psnr), 2),
        'correlation': round(corr, 6),
        'mean_parameter_relative_error_pct': round(float(mre), 4),
        'parameter_estimates': pe,
        'runtime_seconds': round(total_runtime, 1),
        'lmax': lmax,
        'method': 'mcmc_camb_cobaya_framework'
    }
    
    path = os.path.join(results_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"      PSNR={psnr:.2f}dB  corr={corr:.6f}  mean_rel_err={mre:.4f}%")
    for p, v in pe.items():
        print(f"        {p}: true={v['true']} est={v['estimated']} "
              f"err={v['relative_error_pct']:.4f}% σ={v['std']}")
    
    print("[5/5] Creating visualization ...")
    fig = plt.figure(figsize=(18, 14))
    ll = ells[lmin:lmax + 1]
    
    ax = fig.add_subplot(221)
    ax.plot(ll, Dl_obs[lmin:lmax + 1], '.', color='gray', alpha=0.2, ms=1,
            rasterized=True, label='Observed (noisy)')
    ax.plot(ll, Dl_true[lmin:lmax + 1], 'b-', lw=1.5, alpha=.85, label='True')
    ax.plot(ll, Dl_recon[lmin:lmax + 1], 'r--', lw=1.5, alpha=.85, label='MCMC median')
    ax.set(xlabel=r'$\ell$', ylabel=r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]',
           xlim=(lmin, lmax))
    ax.set_title('CMB TT Power Spectrum', fontweight='bold')
    ax.legend(fontsize=10)
    
    ax = fig.add_subplot(222)
    rp = (Dl_recon[lmin:lmax + 1] - Dl_true[lmin:lmax + 1]) / (np.abs(Dl_true[lmin:lmax + 1]) + 1e-10) * 100
    ax.plot(ll, rp, 'g-', alpha=.4, lw=.5)
    w = 15
    sm = np.convolve(rp, np.ones(w) / w, 'valid')
    ax.plot(ll[w // 2:w // 2 + len(sm)], sm, 'r-', lw=1.5, label=f'Smoothed (w={w})')
    ax.axhline(0, color='k', ls='--', alpha=.5)
    ax.set(xlabel=r'$\ell$', ylabel='Residual (%)', ylim=(-5, 5))
    ax.set_title('Recovered − True', fontweight='bold')
    ax.legend()
    
    ax = fig.add_subplot(223)
    pulls = [(pr[p]['median'] - pr[p]['true']) / max(pr[p]['std'], 1e-15) for p in PARAM_NAMES]
    cols = ['#2196F3' if abs(x) < 1 else '#FF9800' if abs(x) < 2 else '#F44336' for x in pulls]
    ax.bar(range(5), pulls, color=cols, alpha=.8, ec='k', lw=.5)
    ax.axhline(0, color='k', lw=.8)
    for y in [-2, -1, 1, 2]:
        ax.axhline(y, color='gray', ls='--' if abs(y) == 1 else ':', alpha=.4)
    ax.set_xticks(range(5))
    ax.set_xticklabels(PARAM_LABELS, fontsize=11)
    ax.set(ylabel=r'Pull $(\hat\theta-\theta_{\rm true})/\sigma$', ylim=(-3.5, 3.5))
    ax.set_title('Parameter Recovery', fontweight='bold')
    
    ax = fig.add_subplot(224)
    ax.axis('off')
    td = []
    for p, lb in zip(PARAM_NAMES, PARAM_LABELS):
        r = pr[p]
        e = metrics['parameter_estimates'][p]
        td.append([lb, f"{r['true']:.5g}", f"{r['median']:.5g}",
                   f"±{r['std']:.4g}", f"{e['relative_error_pct']:.3f}%"])
    tb = ax.table(cellText=td, colLabels=['Param', 'True', 'Median', 'σ', 'Rel.Err'],
                  cellLoc='center', loc='center',
                  colWidths=[.22, .18, .18, .18, .18])
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 1.6)
    for j in range(5):
        tb[0, j].set_facecolor('#E3F2FD')
        tb[0, j].set_text_props(fontweight='bold')
    ax.text(.5, .08,
            f"PSNR={metrics['psnr_dB']:.2f}dB | Corr={metrics['correlation']:.6f} | "
            f"MeanRelErr={metrics['mean_parameter_relative_error_pct']:.4f}%\n"
            f"MCMC+CAMB | ℓmax={lmax} | Runtime={metrics['runtime_seconds']:.0f}s",
            ha='center', va='center', fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=.5', fc='lightyellow', alpha=.8))
    ax.set_title('Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Cobaya: Cosmological Parameter Estimation from CMB Power Spectrum',
                 fontsize=15, fontweight='bold', y=.98)
    plt.tight_layout(rect=[0, 0, 1, .96])
    plot_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved {plot_path}")
    
    np.save(os.path.join(results_dir, "gt_output.npy"), Dl_true)
    np.save(os.path.join(results_dir, "recon_output.npy"), Dl_recon)
    np.save(os.path.join(results_dir, "observed_data.npy"), Dl_obs)
    
    return metrics


# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = ['/data/yjh/cobaya_cosmo_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer and inner data files
    outer_paths = []
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_paths.append(p)

    if not outer_paths:
        print("ERROR: No outer (primary) data file found.")
        sys.exit(1)

    # Load outer data
    outer_path = outer_paths[0]
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")

    args = outer_data['args']
    kwargs = outer_data['kwargs']
    std_output = outer_data['output']

    # Print info about the inputs
    print(f"Number of positional args: {len(args)}")
    print(f"Keyword args keys: {list(kwargs.keys())}")

    # Run the agent's function
    print("\n=== Running agent's run_inversion ===")
    try:
        t_start = time.time()
        agent_output = run_inversion(*args, **kwargs)
        t_agent = time.time() - t_start
        print(f"Agent run_inversion completed in {t_agent:.1f}s")
    except Exception as e:
        print(f"ERROR: Agent's run_inversion raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Check if there are inner data files (chained execution)
    if inner_paths:
        print(f"\nFound {len(inner_paths)} inner data file(s) - Chained execution mode")
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data['args']
        inner_kwargs = inner_data['kwargs']
        std_result = inner_data['output']

        print("Running agent_output as callable with inner data...")
        try:
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Chained call failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nDirect execution mode")
        final_agent_result = agent_output
        std_result = std_output

    # Now evaluate both results using evaluate_results
    # We need the 'data' dict (first argument to run_inversion)
    # Extract the 'data' argument from the function call
    input_data = args[0] if len(args) > 0 else kwargs.get('data', None)
    if input_data is None:
        print("ERROR: Could not extract 'data' argument for evaluation")
        sys.exit(1)

    results_dir_agent = '/tmp/eval_agent'
    results_dir_std = '/tmp/eval_std'

    print("\n=== Evaluating Agent's output ===")
    try:
        agent_runtime = final_agent_result.get('runtime', t_agent) if isinstance(final_agent_result, dict) else t_agent
        metrics_agent = evaluate_results(input_data, final_agent_result, agent_runtime, results_dir_agent)
    except Exception as e:
        print(f"ERROR: evaluate_results failed on agent output: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Evaluating Standard output ===")
    try:
        std_runtime = std_result.get('runtime', 0.0) if isinstance(std_result, dict) else 0.0
        metrics_std = evaluate_results(input_data, std_result, std_runtime, results_dir_std)
    except Exception as e:
        print(f"ERROR: evaluate_results failed on standard output: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract key metrics for comparison
    psnr_agent = metrics_agent['psnr_dB']
    psnr_std = metrics_std['psnr_dB']
    corr_agent = metrics_agent['correlation']
    corr_std = metrics_std['correlation']
    mre_agent = metrics_agent['mean_parameter_relative_error_pct']
    mre_std = metrics_std['mean_parameter_relative_error_pct']

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"PSNR (dB)       -> Agent: {psnr_agent:.2f}, Standard: {psnr_std:.2f}")
    print(f"Correlation     -> Agent: {corr_agent:.6f}, Standard: {corr_std:.6f}")
    print(f"Mean Rel Err (%)-> Agent: {mre_agent:.4f}, Standard: {mre_std:.4f}")
    print("=" * 60)

    # Determine pass/fail
    # PSNR: Higher is better (allow 10% margin below standard)
    # Correlation: Higher is better (allow small margin)
    # Mean Relative Error: Lower is better (allow 50% margin above standard due to MCMC stochasticity)
    
    passed = True
    reasons = []

    # PSNR check: agent should not be drastically worse
    if psnr_std > 0 and psnr_agent < psnr_std * 0.8:
        passed = False
        reasons.append(f"PSNR too low: {psnr_agent:.2f} vs {psnr_std:.2f} (threshold: {psnr_std * 0.8:.2f})")

    # Correlation check
    if corr_agent < corr_std * 0.95:
        passed = False
        reasons.append(f"Correlation too low: {corr_agent:.6f} vs {corr_std:.6f}")

    # Mean relative error check: allow generous margin since MCMC is stochastic
    # Agent's error should not be more than 3x the standard error (or 5% absolute max)
    mre_threshold = max(mre_std * 3.0, 5.0)
    if mre_agent > mre_threshold:
        passed = False
        reasons.append(f"Mean relative error too high: {mre_agent:.4f}% vs threshold {mre_threshold:.4f}%")

    # Also check that the agent produced valid output structure
    required_keys = ['parameter_results', 'posterior_samples', 'Dl_recon', 'runtime']
    for key in required_keys:
        if key not in final_agent_result:
            passed = False
            reasons.append(f"Missing key in output: {key}")

    if passed:
        print("\n✅ TEST PASSED: Agent's run_inversion performance is acceptable.")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED:")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == '__main__':
    main()