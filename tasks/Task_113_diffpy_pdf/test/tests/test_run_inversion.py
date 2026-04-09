import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the agent's function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the evaluate_results function verbatim from Reference B
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_113_diffpy_pdf"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def evaluate_results(r, G_gt, G_noisy, G_fit, params_true, a_fit, B_fit, scale_fit):
    """
    Evaluate reconstruction quality and save results.
    """
    a_true = params_true['a']
    B_true = params_true['B']
    scale_true = params_true['scale']

    g_max = np.max(np.abs(G_gt)) + 1e-12
    gt_n = G_gt / g_max
    fi_n = G_fit / g_max

    mse = np.mean((gt_n - fi_n)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    g = gt_n - np.mean(gt_n)
    f = fi_n - np.mean(fi_n)
    cc = np.sum(g * f) / (np.sqrt(np.sum(g**2) * np.sum(f**2)) + 1e-12)

    a_err = abs(a_fit - a_true) / a_true * 100
    B_err = abs(B_fit - B_true) / B_true * 100
    scale_err = abs(scale_fit - scale_true) / scale_true * 100

    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "lattice_constant_true": float(a_true),
        "lattice_constant_fitted": float(a_fit),
        "lattice_constant_error_pct": float(a_err),
        "debye_waller_true": float(B_true),
        "debye_waller_fitted": float(B_fit),
        "debye_waller_error_pct": float(B_err),
        "scale_true": float(scale_true),
        "scale_fitted": float(scale_fit),
        "scale_error_pct": float(scale_err),
        "RMSE": float(np.sqrt(mse)),
    }

    r_min = r[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(r, G_gt, "b-", linewidth=1.5, label="Ground Truth G(r)")
    axes[0, 0].plot(r, G_fit, "r--", linewidth=1.5, label="Fitted G(r)")
    axes[0, 0].set_xlabel("r (Å)", fontsize=12)
    axes[0, 0].set_ylabel("G(r)", fontsize=12)
    axes[0, 0].set_title("PDF: Ground Truth vs Fit", fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_xlim([r_min, 15])

    axes[0, 1].plot(r, G_noisy, "gray", alpha=0.6, linewidth=0.8, label="Noisy data")
    axes[0, 1].plot(r, G_fit, "r-", linewidth=1.5, label="Fitted G(r)")
    axes[0, 1].set_xlabel("r (Å)", fontsize=12)
    axes[0, 1].set_ylabel("G(r)", fontsize=12)
    axes[0, 1].set_title("Noisy Data vs Fit", fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_xlim([r_min, 15])

    residual = G_gt - G_fit
    axes[1, 0].plot(r, residual, "g-", linewidth=1.0)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("r (Å)", fontsize=12)
    axes[1, 0].set_ylabel("Residual", fontsize=12)
    axes[1, 0].set_title(f"Residual (RMSE = {metrics['RMSE']:.4f})", fontsize=14)
    axes[1, 0].set_xlim([r_min, 15])

    axes[1, 1].axis("off")
    table_data = [
        ["Parameter", "True", "Fitted", "Error (%)"],
        ["a (Å)", f"{metrics['lattice_constant_true']:.4f}",
         f"{metrics['lattice_constant_fitted']:.4f}",
         f"{metrics['lattice_constant_error_pct']:.2f}%"],
        ["B (Å²)", f"{metrics['debye_waller_true']:.4f}",
         f"{metrics['debye_waller_fitted']:.4f}",
         f"{metrics['debye_waller_error_pct']:.2f}%"],
        ["Scale", f"{metrics['scale_true']:.4f}",
         f"{metrics['scale_fitted']:.4f}",
         f"{metrics['scale_error_pct']:.2f}%"],
        ["", "", "", ""],
        ["PSNR", f"{metrics['PSNR']:.2f} dB", "", ""],
        ["CC", f"{metrics['CC']:.4f}", "", ""],
    ]
    table = axes[1, 1].table(
        cellText=table_data, loc="center", cellLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Fitted Parameters", fontsize=14)

    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()

    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), G_gt)
        np.save(os.path.join(d, "recon_output.npy"), G_fit)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ============================================================
# We also need the forward_operator to generate ground truth G(r)
# ============================================================
def fcc_neighbor_distances(a, r_max, max_shell=200):
    distances = []
    n_max = int(np.ceil(r_max / a)) + 1
    for h in range(-n_max, n_max + 1):
        for k in range(-n_max, n_max + 1):
            for l in range(-n_max, n_max + 1):
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (h + bx) * a
                    y = (k + by) * a
                    z = (l + bz) * a
                    d = np.sqrt(x**2 + y**2 + z**2)
                    if 0.1 < d < r_max:
                        distances.append(d)
    distances = np.sort(distances)
    shells = []
    tol = 0.01
    i = 0
    while i < len(distances) and len(shells) < max_shell:
        d_ref = distances[i]
        count = 0
        while i < len(distances) and abs(distances[i] - d_ref) < tol:
            count += 1
            i += 1
        shells.append((d_ref, count))
    return shells


def forward_operator(r, a, B, scale, r_max):
    shells = fcc_neighbor_distances(a, r_max)
    sigma = np.sqrt(B)
    G = np.zeros_like(r)
    rho0 = 4 / a**3
    for d_n, coord_n in shells:
        sigma_n = sigma * np.sqrt(1 + 0.002 * d_n**2)
        amplitude = coord_n / (4 * np.pi * d_n**2 * rho0)
        peak = amplitude * np.exp(-0.5 * ((r - d_n) / sigma_n)**2) / (sigma_n * np.sqrt(2 * np.pi))
        G += peak
    G = scale * G / (np.max(np.abs(G)) + 1e-12)
    G *= np.exp(-0.01 * r**2)
    return G


# ============================================================
# Main Test Logic
# ============================================================
def main():
    data_paths = ['/data/yjh/diffpy_pdf_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Classify files
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
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")

    args = outer_data['args']
    kwargs = outer_data['kwargs']
    std_output = outer_data['output']

    # Print input info
    print(f"Number of positional args: {len(args)}")
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={a.shape}, dtype={a.dtype}")
        else:
            print(f"  arg[{i}]: {type(a).__name__} = {a}")
    print(f"Kwargs: {list(kwargs.keys())}")

    # Extract inputs
    r = args[0]
    G_measured = args[1]
    r_max = args[2]

    # ============================================================
    # Pattern 1: Direct Execution
    # ============================================================
    if len(inner_paths) == 0:
        print("\n=== Pattern 1: Direct Execution ===")

        # Run agent's function
        print("Running agent's run_inversion...")
        try:
            agent_output = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR: Agent run_inversion failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Extract results
        # Output: (a_fit, B_fit, scale_fit, G_fit, result)
        a_fit_agent = agent_output[0]
        B_fit_agent = agent_output[1]
        scale_fit_agent = agent_output[2]
        G_fit_agent = agent_output[3]

        a_fit_std = std_output[0]
        B_fit_std = std_output[1]
        scale_fit_std = std_output[2]
        G_fit_std = std_output[3]

        print(f"\nAgent fitted params: a={a_fit_agent:.6f}, B={B_fit_agent:.6f}, scale={scale_fit_agent:.6f}")
        print(f"Std fitted params:   a={a_fit_std:.6f}, B={B_fit_std:.6f}, scale={scale_fit_std:.6f}")

        # We need to reconstruct the ground truth G(r) and params_true for evaluate_results.
        # The ground truth params are embedded in the gen_data code. 
        # From the gen_data code, the standard approach uses known true parameters.
        # We need to figure out the true parameters. Since this is an inversion/fitting problem,
        # the "true" parameters generated the noisy data. We can infer them from the standard output
        # or we can try to use reasonable values.
        
        # The standard output gives us the fitted values from the reference implementation.
        # For evaluation, we need: G_gt (ground truth without noise), G_noisy, and true params.
        # 
        # Since we don't have direct access to the true params, we'll use the standard fitted
        # params as "true" for relative comparison, OR we can compute G_gt from std params.
        #
        # Better approach: use the std fitted params as "true" parameters for evaluation,
        # since both agent and std should converge to the same solution.
        # Actually, let's compute G_gt using the std fitted parameters as ground truth.

        # Construct params_true from std output (best available reference)
        params_true = {
            'a': float(a_fit_std),
            'B': float(B_fit_std),
            'scale': float(scale_fit_std),
        }

        # G_gt from standard fitted params
        G_gt = G_fit_std

        # Evaluate agent
        print("\n--- Evaluating Agent Output ---")
        try:
            metrics_agent = evaluate_results(
                r, G_gt, G_measured, G_fit_agent, params_true,
                a_fit_agent, B_fit_agent, scale_fit_agent
            )
        except Exception as e:
            print(f"ERROR evaluating agent results: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Evaluate standard (self-comparison, should be perfect)
        print("\n--- Evaluating Standard Output ---")
        try:
            metrics_std = evaluate_results(
                r, G_gt, G_measured, G_fit_std, params_true,
                a_fit_std, B_fit_std, scale_fit_std
            )
        except Exception as e:
            print(f"ERROR evaluating standard results: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Report
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Agent PSNR: {metrics_agent['PSNR']:.4f} dB")
        print(f"Std    PSNR: {metrics_std['PSNR']:.4f} dB")
        print(f"Agent CC:   {metrics_agent['CC']:.6f}")
        print(f"Std    CC:   {metrics_std['CC']:.6f}")
        print(f"Agent RMSE: {metrics_agent['RMSE']:.6f}")
        print(f"Std    RMSE: {metrics_std['RMSE']:.6f}")
        print(f"Agent lattice error: {metrics_agent['lattice_constant_error_pct']:.4f}%")
        print(f"Agent DW error:      {metrics_agent['debye_waller_error_pct']:.4f}%")
        print(f"Agent scale error:   {metrics_agent['scale_error_pct']:.4f}%")

        # Determine pass/fail
        # PSNR and CC are "higher is better"
        # We use CC as primary metric since the std self-comparison will have CC=1.0
        # For a fair comparison, check if agent's CC is reasonably high (> 0.9)
        # and agent's parameter errors are small
        
        psnr_agent = metrics_agent['PSNR']
        cc_agent = metrics_agent['CC']
        
        # Also do a direct comparison of fitted parameters
        a_diff = abs(a_fit_agent - a_fit_std) / (abs(a_fit_std) + 1e-12)
        B_diff = abs(B_fit_agent - B_fit_std) / (abs(B_fit_std) + 1e-12)
        scale_diff = abs(scale_fit_agent - scale_fit_std) / (abs(scale_fit_std) + 1e-12)

        print(f"\nRelative parameter differences (agent vs std):")
        print(f"  a:     {a_diff*100:.4f}%")
        print(f"  B:     {B_diff*100:.4f}%")
        print(f"  scale: {scale_diff*100:.4f}%")

        # Pass criteria:
        # 1. CC >= 0.90 (agent's fit correlates well with std fit)
        # 2. Parameter differences < 10%
        passed = True
        reasons = []

        if cc_agent < 0.90:
            passed = False
            reasons.append(f"CC too low: {cc_agent:.4f} < 0.90")

        if a_diff > 0.10:
            passed = False
            reasons.append(f"Lattice constant differs by {a_diff*100:.2f}% (>10%)")

        if B_diff > 0.50:
            # B (Debye-Waller) can be harder to fit precisely, allow 50%
            passed = False
            reasons.append(f"Debye-Waller differs by {B_diff*100:.2f}% (>50%)")

        if scale_diff > 0.50:
            passed = False
            reasons.append(f"Scale differs by {scale_diff*100:.2f}% (>50%)")

        print("\n" + "=" * 60)
        if passed:
            print("TEST PASSED: Agent performance is acceptable.")
            print(f"Scores -> Agent PSNR: {psnr_agent:.2f}, CC: {cc_agent:.4f}")
            sys.exit(0)
        else:
            print("TEST FAILED: Agent performance degraded.")
            for r in reasons:
                print(f"  - {r}")
            sys.exit(1)

    else:
        # ============================================================
        # Pattern 2: Chained Execution
        # ============================================================
        print("\n=== Pattern 2: Chained Execution ===")

        # Run outer function
        print("Running agent's run_inversion (outer)...")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR: Agent run_inversion failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Load and run inner data
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)

            inner_args = inner_data['args']
            inner_kwargs = inner_data['kwargs']
            inner_std_output = inner_data['output']

            print("Running agent operator (inner)...")
            try:
                agent_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            print(f"Agent inner result type: {type(agent_result)}")
            print(f"Std inner result type: {type(inner_std_output)}")

            # Basic comparison
            print("Chained execution completed successfully.")

        sys.exit(0)


if __name__ == "__main__":
    main()