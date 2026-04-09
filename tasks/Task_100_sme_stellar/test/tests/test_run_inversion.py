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
ASSETS_DIR = "/data/yjh/website_assets/Task_100_sme_stellar"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def evaluate_results(data_dict, result_dict, results_dir, assets_dir):
    """
    Evaluate results: compute metrics, save outputs, and generate visualizations.
    """
    wavelength = data_dict["wavelength"]
    flux_gt = data_dict["flux_gt"]
    flux_obs = data_dict["flux_obs"]
    gt_params = data_dict["gt_params"]

    flux_fit = result_dict["flux_fit"]
    fit_params = result_dict["fit_params"]

    gt_teff, gt_logg, gt_feh, gt_abundances = gt_params
    fit_teff, fit_logg, fit_feh, fit_abundances = fit_params

    # Compute spectrum metrics
    mse = np.mean((flux_gt - flux_fit) ** 2)
    psnr = 10.0 * np.log10(flux_gt.max() ** 2 / mse) if mse > 0 else 100.0
    cc = float(np.corrcoef(flux_gt.flatten(), flux_fit.flatten())[0, 1])

    # Compute parameter relative errors
    param_errors = {}

    # T_eff relative error
    if abs(gt_teff) > 1e-6:
        param_errors["RE_T_eff"] = abs(fit_teff - gt_teff) / abs(gt_teff)
    else:
        param_errors["RE_T_eff"] = abs(fit_teff - gt_teff)

    # log_g relative error
    if abs(gt_logg) > 1e-6:
        param_errors["RE_log_g"] = abs(fit_logg - gt_logg) / abs(gt_logg)
    else:
        param_errors["RE_log_g"] = abs(fit_logg - gt_logg)

    # [Fe/H] - use absolute error since it can be zero
    param_errors["RE_[Fe/H]"] = abs(fit_feh - gt_feh)

    # Abundance absolute errors
    for elem in sorted(gt_abundances.keys()):
        param_errors[f"AE_{elem}"] = abs(fit_abundances[elem] - gt_abundances[elem])

    # Build metrics dictionary
    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RE_Teff": float(param_errors["RE_T_eff"]),
        "RE_logg": float(param_errors["RE_log_g"]),
    }
    for elem in sorted(gt_abundances.keys()):
        metrics[f"AE_{elem}"] = float(param_errors[f"AE_{elem}"])

    # Print results
    print(f"    Spectrum PSNR = {psnr:.2f} dB")
    print(f"    Spectrum CC   = {cc:.6f}")
    print(
        f"    T_eff: GT={gt_teff:.0f} K, Fit={fit_teff:.0f} K, RE={param_errors['RE_T_eff']:.4f}"
    )
    print(
        f"    log_g: GT={gt_logg:.2f}, Fit={fit_logg:.2f}, RE={param_errors['RE_log_g']:.4f}"
    )
    print(f"    [Fe/H]: GT={gt_feh:.2f}, Fit={fit_feh:.2f}")
    for elem in sorted(gt_abundances.keys()):
        print(
            f"    [{elem}/H]: GT={gt_abundances[elem]:.2f}, Fit={fit_abundances[elem]:.2f}, AE={param_errors[f'AE_{elem}']:.4f}"
        )

    # Save outputs
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), flux_gt)
        np.save(os.path.join(d, "recon_output.npy"), flux_fit)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: full spectrum overlay
    ax = axes[0, 0]
    ax.plot(wavelength, flux_obs, "gray", alpha=0.4, lw=0.5, label="Observed")
    ax.plot(wavelength, flux_gt, "b-", lw=1.0, label="GT spectrum")
    ax.plot(wavelength, flux_fit, "r--", lw=1.0, label="Fitted spectrum")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title("Stellar Spectrum: GT vs Fitted")
    ax.legend(fontsize=8)

    # Panel 2: zoom on Na D lines
    ax = axes[0, 1]
    mask = (wavelength > 5870) & (wavelength < 5920)
    ax.plot(
        wavelength[mask],
        flux_obs[mask],
        "gray",
        alpha=0.5,
        lw=0.8,
        label="Observed",
    )
    ax.plot(wavelength[mask], flux_gt[mask], "b-", lw=1.2, label="GT")
    ax.plot(wavelength[mask], flux_fit[mask], "r--", lw=1.2, label="Fitted")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title("Zoom: Na D doublet (5890/5896 Å)")
    ax.legend(fontsize=8)

    # Panel 3: residuals
    ax = axes[1, 0]
    residual = flux_gt - flux_fit
    ax.plot(wavelength, residual, "k-", lw=0.5)
    ax.axhline(0, color="r", ls="--", lw=0.5)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Residual (GT - Fit)")
    ax.set_title(f"Residuals | PSNR={psnr:.1f} dB, CC={cc:.4f}")

    # Panel 4: parameter comparison bar chart
    ax = axes[1, 1]
    labels = ["T_eff/1000", "log_g", "[Fe/H]+1"]
    gt_v = [gt_teff / 1000, gt_logg, gt_feh + 1]
    fit_v = [fit_teff / 1000, fit_logg, fit_feh + 1]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, gt_v, 0.3, label="GT", color="steelblue")
    ax.bar(x + 0.15, fit_v, 0.3, label="Fitted", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Stellar Parameters")
    ax.legend()

    plt.tight_layout()
    for path in [
        os.path.join(results_dir, "vis_result.png"),
        os.path.join(assets_dir, "vis_result.png"),
    ]:
        fig.savefig(path, dpi=150)
    plt.close(fig)

    return metrics


# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = [
        "/data/yjh/sme_stellar_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl"
    ]

    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if "parent_function" in basename or "parent_" in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, "rb") as f:
        outer_data = dill.load(f)

    outer_args = outer_data.get("args", ())
    outer_kwargs = outer_data.get("kwargs", {})
    std_output = outer_data.get("output", None)

    print(f"Outer function: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}, Number of kwargs: {len(outer_kwargs)}")

    if len(inner_paths) > 0:
        # ---- Pattern 2: Chained Execution ----
        print("Detected CHAINED execution pattern.")

        # Run outer function to get operator
        print("Running run_inversion (outer) to get operator...")
        try:
            agent_operator = run_inversion(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR running outer function: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, "rb") as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get("args", ())
        inner_kwargs = inner_data.get("kwargs", {})
        std_result = inner_data.get("output", None)

        print("Running operator (inner)...")
        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # ---- Pattern 1: Direct Execution ----
        print("Detected DIRECT execution pattern.")

        print("Running run_inversion...")
        try:
            agent_result = run_inversion(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)

        std_result = std_output

    # ============================================================
    # Evaluation Phase
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)

    # We need data_dict for evaluation. It's the first arg to run_inversion.
    data_dict = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get("data_dict", None)

    if data_dict is None:
        print("ERROR: Could not extract data_dict for evaluation.")
        sys.exit(1)

    # Create separate results directories for agent and standard
    agent_results_dir = os.path.join(RESULTS_DIR, "agent")
    std_results_dir = os.path.join(RESULTS_DIR, "standard")
    agent_assets_dir = os.path.join(ASSETS_DIR, "agent")
    std_assets_dir = os.path.join(ASSETS_DIR, "standard")
    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(std_results_dir, exist_ok=True)
    os.makedirs(agent_assets_dir, exist_ok=True)
    os.makedirs(std_assets_dir, exist_ok=True)

    # Evaluate agent result
    print("\n--- Agent Results ---")
    try:
        metrics_agent = evaluate_results(data_dict, agent_result, agent_results_dir, agent_assets_dir)
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard result
    print("\n--- Standard Results ---")
    try:
        metrics_std = evaluate_results(data_dict, std_result, std_results_dir, std_assets_dir)
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Comparison and Verdict
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    psnr_agent = metrics_agent["PSNR"]
    psnr_std = metrics_std["PSNR"]
    cc_agent = metrics_agent["CC"]
    cc_std = metrics_std["CC"]
    re_teff_agent = metrics_agent["RE_Teff"]
    re_teff_std = metrics_std["RE_Teff"]
    re_logg_agent = metrics_agent["RE_logg"]
    re_logg_std = metrics_std["RE_logg"]

    print(f"PSNR  -> Agent: {psnr_agent:.2f} dB, Standard: {psnr_std:.2f} dB")
    print(f"CC    -> Agent: {cc_agent:.6f}, Standard: {cc_std:.6f}")
    print(f"RE_Teff -> Agent: {re_teff_agent:.6f}, Standard: {re_teff_std:.6f}")
    print(f"RE_logg -> Agent: {re_logg_agent:.6f}, Standard: {re_logg_std:.6f}")

    # Print all abundance errors
    for key in sorted(metrics_agent.keys()):
        if key.startswith("AE_"):
            print(
                f"{key} -> Agent: {metrics_agent[key]:.6f}, Standard: {metrics_std[key]:.6f}"
            )

    # ============================================================
    # Determine pass/fail
    # ============================================================
    # Primary metric: PSNR (higher is better) and CC (higher is better)
    # Allow 10% margin for PSNR degradation
    # For relative errors, agent can be up to 2x worse (since optimization is stochastic)

    passed = True
    reasons = []

    # PSNR check: agent should be at least 90% of standard
    if psnr_agent < psnr_std * 0.90:
        passed = False
        reasons.append(
            f"PSNR degraded: agent={psnr_agent:.2f} < 90% of std={psnr_std:.2f}"
        )

    # CC check: agent should be close to standard
    if cc_agent < cc_std * 0.95:
        passed = False
        reasons.append(
            f"CC degraded: agent={cc_agent:.6f} < 95% of std={cc_std:.6f}"
        )

    # RE_Teff check: allow up to 3x the standard error (stochastic optimizer)
    if re_teff_std > 1e-8:
        if re_teff_agent > re_teff_std * 3.0 and re_teff_agent > 0.05:
            passed = False
            reasons.append(
                f"RE_Teff too large: agent={re_teff_agent:.6f} > 3x std={re_teff_std:.6f}"
            )
    else:
        if re_teff_agent > 0.05:
            passed = False
            reasons.append(f"RE_Teff too large: agent={re_teff_agent:.6f}")

    # RE_logg check
    if re_logg_std > 1e-8:
        if re_logg_agent > re_logg_std * 3.0 and re_logg_agent > 0.05:
            passed = False
            reasons.append(
                f"RE_logg too large: agent={re_logg_agent:.6f} > 3x std={re_logg_std:.6f}"
            )
    else:
        if re_logg_agent > 0.05:
            passed = False
            reasons.append(f"RE_logg too large: agent={re_logg_agent:.6f}")

    print("\n" + "=" * 60)
    if passed:
        print("VERDICT: PASS - Agent performance is acceptable.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("VERDICT: FAIL - Agent performance degraded significantly.")
        for r in reasons:
            print(f"  - {r}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()