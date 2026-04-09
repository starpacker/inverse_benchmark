import os

import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def evaluate_results(results, out_dir):
    """
    Evaluate and visualize inversion results.
    
    Parameters
    ----------
    results : list of dict
        Output from run_inversion
    out_dir : str
        Output directory for saving results
    
    Returns
    -------
    metrics : dict
        Aggregated metrics including per-case and overall statistics
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # ── Metrics ──
    metrics = {"test_cases": []}
    all_re = {"T_e": [], "n_e": [], "V_p": [], "I_ion_sat": [], "V_f": []}
    
    for r in results:
        case_metric = {
            "case": r["case"],
            "true": {
                "T_e": r["true_params"]["T_e"],
                "n_e": r["true_params"]["n_e"],
                "V_p": r["true_params"]["V_p"],
                "I_ion_sat": r["true_params"]["I_ion_sat"],
                "V_f": r["V_f_true"],
            },
            "estimated": {
                "T_e": r["fitted_params"]["T_e"],
                "n_e": r["fitted_params"]["n_e"],
                "V_p": r["fitted_params"]["V_p"],
                "I_ion_sat": r["fitted_params"]["I_ion_sat"],
                "V_f": r["fitted_params"]["V_f"],
            },
            "relative_errors": {k: f"{v*100:.4f}%" for k, v in r["relative_errors"].items()},
        }
        metrics["test_cases"].append(case_metric)
        
        for k in all_re:
            all_re[k].append(r["relative_errors"][k])
        
        print(f"\n--- Case: {r['case']} ---")
        print(f"  T_e : true={r['true_params']['T_e']:.2f} eV,  est={r['fitted_params']['T_e']:.4f} eV,  RE={r['relative_errors']['T_e']*100:.4f}%")
        print(f"  n_e : true={r['true_params']['n_e']:.2e} m⁻³, est={r['fitted_params']['n_e']:.4e} m⁻³, RE={r['relative_errors']['n_e']*100:.4f}%")
        print(f"  V_p : true={r['true_params']['V_p']:.2f} V,   est={r['fitted_params']['V_p']:.4f} V,   RE={r['relative_errors']['V_p']*100:.4f}%")
        print(f"  V_f : true={r['V_f_true']:.4f} V,   est={r['fitted_params']['V_f']:.4f} V,   RE={r['relative_errors']['V_f']*100:.4f}%")
    
    mean_re = {k: np.mean(v) * 100 for k, v in all_re.items()}
    metrics["mean_relative_errors"] = {k: f"{v:.4f}%" for k, v in mean_re.items()}
    metrics["overall_mean_RE"] = f"{np.mean(list(mean_re.values())):.4f}%"
    
    print(f"\n{'='*60}")
    print("Mean Relative Errors across all cases:")
    for k, v in mean_re.items():
        status = "✓ PASS" if v < 10 else "✗ FAIL"
        print(f"  {k:12s}: {v:.4f}%  {status}")
    overall = np.mean(list(mean_re.values()))
    print(f"  {'Overall':12s}: {overall:.4f}%  {'✓ PASS' if overall < 10 else '✗ FAIL'}")
    
    # ── Save metrics ──
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[INFO] Saved metrics → {metrics_path}")
    
    # Ground truth: dict of all true parameters for baseline case
    gt = {
        "T_e": results[0]["true_params"]["T_e"],
        "n_e": results[0]["true_params"]["n_e"],
        "V_p": results[0]["true_params"]["V_p"],
        "I_ion_sat": results[0]["true_params"]["I_ion_sat"],
        "V_f": results[0]["V_f_true"],
        "I_V_data": {
            "V": results[0]["V"].tolist(),
            "I_clean": results[0]["I_clean"].tolist(),
            "I_noisy": results[0]["I_noisy"].tolist(),
        },
    }
    gt_path = os.path.join(out_dir, "ground_truth.npy")
    np.save(gt_path, gt, allow_pickle=True)
    print(f"[INFO] Saved ground truth → {gt_path}")
    
    # Reconstruction: dict of fitted parameters for baseline case
    recon = {
        "T_e": results[0]["fitted_params"]["T_e"],
        "n_e": results[0]["fitted_params"]["n_e"],
        "V_p": results[0]["fitted_params"]["V_p"],
        "I_ion_sat": results[0]["fitted_params"]["I_ion_sat"],
        "V_f": results[0]["fitted_params"]["V_f"],
        "I_V_data": {
            "V": results[0]["V"].tolist(),
            "I_fitted": results[0]["I_fitted"].tolist(),
        },
    }
    recon_path = os.path.join(out_dir, "reconstruction.npy")
    np.save(recon_path, recon, allow_pickle=True)
    print(f"[INFO] Saved reconstruction → {recon_path}")
    
    # ── Visualization ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 157: Langmuir Probe I-V Curve Inversion", fontsize=14, fontweight="bold")
    
    # Use baseline case (index 0) for panels 1-3
    r = results[0]
    V = r["V"]
    I_noisy = r["I_noisy"]
    I_clean = r["I_clean"]
    I_fitted = r["I_fitted"]
    
    # Panel 1: Noisy I-V data
    ax = axes[0, 0]
    ax.scatter(V, I_noisy * 1e3, s=2, alpha=0.5, color="steelblue", label="Noisy data")
    ax.set_xlabel("Bias Voltage V [V]")
    ax.set_ylabel("Current I [mA]")
    ax.set_title("(a) Noisy I-V Data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: True vs Fitted curves
    ax = axes[0, 1]
    ax.plot(V, I_clean * 1e3, "k-", lw=2, label="Ground truth")
    ax.plot(V, I_fitted * 1e3, "r--", lw=2, label="Fitted")
    ax.scatter(V, I_noisy * 1e3, s=1, alpha=0.3, color="gray", label="Noisy data")
    ax.set_xlabel("Bias Voltage V [V]")
    ax.set_ylabel("Current I [mA]")
    ax.set_title("(b) True vs Fitted I-V Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Residuals
    ax = axes[1, 0]
    residuals = (I_fitted - I_clean) * 1e3
    ax.plot(V, residuals, "g-", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Bias Voltage V [V]")
    ax.set_ylabel("Residual [mA]")
    ax.set_title("(c) Fit Residuals (Fitted − True)")
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Parameter comparison bar chart (all cases)
    ax = axes[1, 1]
    param_names = ["T_e", "n_e", "V_p", "I_ion_sat"]
    mean_re_plot = {}
    for pn in param_names:
        vals = [r2["relative_errors"][pn] * 100 for r2 in results]
        mean_re_plot[pn] = np.mean(vals)
    x = np.arange(len(param_names))
    colors = ["#2196F3" if v < 5 else "#FF9800" if v < 10 else "#F44336" for v in mean_re_plot.values()]
    bars = ax.bar(x, list(mean_re_plot.values()), color=colors, edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["$T_e$", "$n_e$", "$V_p$", "$I_{ion,sat}$"])
    ax.set_ylabel("Mean Relative Error [%]")
    ax.set_title(f"(d) Mean RE Across {len(results)} Cases")
    ax.axhline(10, color="red", ls="--", lw=1, label="10% threshold")
    for bar, val in zip(bars, mean_re_plot.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "reconstruction_result.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot → {plot_path}")
    
    return metrics
