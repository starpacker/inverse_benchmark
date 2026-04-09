import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

# Import the target function
from agent_run_inversion import run_inversion

# ─── Inject the Referee (evaluate_results) verbatim from Reference B ───

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_116_thermoelastic"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def evaluate_results(gt_stress_sum, recon_stress_sum, R, THETA, delta_T_noisy):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Computes PSNR, SSIM, RMSE, and correlation coefficient metrics,
    saves results to disk, and creates visualization plots.
    
    Parameters
    ----------
    gt_stress_sum : ndarray
        Ground truth stress sum field (Pa)
    recon_stress_sum : ndarray
        Reconstructed stress sum field (Pa)
    R : ndarray
        Radial coordinate meshgrid
    THETA : ndarray
        Angular coordinate meshgrid
    delta_T_noisy : ndarray
        Noisy temperature measurements (K)
    
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, CC, RMSE, RMSE_MPa
    """
    # Compute metrics
    mse = np.mean((gt_stress_sum - recon_stress_sum) ** 2)
    data_range = np.max(gt_stress_sum) - np.min(gt_stress_sum)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))
    
    cc = float(np.corrcoef(gt_stress_sum.ravel(), recon_stress_sum.ravel())[0, 1])
    rmse = float(np.sqrt(mse))
    
    # SSIM — normalize to [0, 1]
    gt_n = (gt_stress_sum - gt_stress_sum.min()) / (gt_stress_sum.max() - gt_stress_sum.min() + 1e-30)
    rc_n = (recon_stress_sum - recon_stress_sum.min()) / (recon_stress_sum.max() - recon_stress_sum.min() + 1e-30)
    ssim_val = float(ssim(gt_n, rc_n, data_range=1.0))
    
    metrics = {
        "PSNR": float(psnr),
        "SSIM": ssim_val,
        "CC": cc,
        "RMSE": rmse,
        "RMSE_MPa": rmse / 1e6
    }
    
    # Print metrics
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  SSIM = {metrics['SSIM']:.4f}")
    print(f"  CC   = {metrics['CC']:.6f}")
    print(f"  RMSE = {metrics['RMSE_MPa']:.4f} MPa")
    
    # Save numerical results
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_stress_sum)
        np.save(os.path.join(d, "recon_output.npy"), recon_stress_sum)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # (a) GT stress sum
    ax = axes[0, 0]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, gt_stress_sum / 1e6, cmap="RdBu_r", shading="auto")
    ax.set_title("GT Stress Sum  σ₁+σ₂  (MPa)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # (b) Temperature change
    ax = axes[0, 1]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, delta_T_noisy * 1e3, cmap="coolwarm", shading="auto")
    ax.set_title("Measured ΔT (mK)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # (c) Recovered stress sum
    ax = axes[1, 0]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, recon_stress_sum / 1e6, cmap="RdBu_r", shading="auto")
    ax.set_title(f"Recovered σ₁+σ₂  (PSNR={metrics['PSNR']:.1f} dB)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # (d) Error
    ax = axes[1, 1]
    err = (gt_stress_sum - recon_stress_sum) / 1e6
    im = ax.pcolormesh(X * 1e3, Y * 1e3, err, cmap="bwr", shading="auto")
    ax.set_title(f"Error  (RMSE={metrics['RMSE_MPa']:.2f} MPa, CC={metrics['CC']:.4f})")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    plt.suptitle("Thermoelastic Stress Analysis — Plate with Hole", fontsize=14, y=1.02)
    plt.tight_layout()
    
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics


# ─── Main test logic ───

def main():
    data_paths = ['/data/yjh/thermoelastic_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer and inner data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # Also scan the directory for any inner files we might have missed
    if outer_path:
        data_dir = os.path.dirname(outer_path)
        if os.path.isdir(data_dir):
            for fname in os.listdir(data_dir):
                full = os.path.join(data_dir, fname)
                if full not in data_paths and 'parent_function' in fname and 'run_inversion' in fname:
                    inner_paths.append(full)

    print(f"Outer data: {outer_path}")
    print(f"Inner data: {inner_paths}")

    # Load outer data
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    func_name = outer_data.get('func_name', 'run_inversion')

    print(f"Function: {func_name}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")

    if len(inner_paths) > 0:
        # ─── Pattern 2: Chained Execution ───
        print("\n=== Pattern 2: Chained Execution ===")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_std_output = inner_data.get('output', None)

            try:
                agent_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR running inner call: {e}")
                traceback.print_exc()
                sys.exit(1)

            std_result = inner_std_output
            # Use these for evaluation
            break  # handle first inner path

    else:
        # ─── Pattern 1: Direct Execution ───
        print("\n=== Pattern 1: Direct Execution ===")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)

        std_result = std_output

    # ─── Evaluation Phase ───
    print("\n=== Evaluation Phase ===")

    # The evaluate_results function needs: gt_stress_sum, recon_stress_sum, R, THETA, delta_T_noisy
    # gt_stress_sum = std_result (ground truth output)
    # recon_stress_sum = agent_result
    # delta_T_noisy = args[0] (the delta_T input)
    # We need R and THETA — these are coordinate meshgrids. We'll create dummy ones
    # matching the shape of the output if not available elsewhere.

    gt_stress_sum = std_result
    recon_stress_sum = agent_result
    delta_T_noisy = args[0] if len(args) > 0 else kwargs.get('delta_T', None)

    print(f"GT shape: {gt_stress_sum.shape if hasattr(gt_stress_sum, 'shape') else 'N/A'}")
    print(f"Agent shape: {recon_stress_sum.shape if hasattr(recon_stress_sum, 'shape') else 'N/A'}")

    # Generate R, THETA grids matching the data shape
    # These are needed for visualization only; they don't affect metric computation
    shape = gt_stress_sum.shape
    nr, ntheta = shape
    r_vals = np.linspace(0.005, 0.05, nr)  # reasonable physical range for plate-with-hole
    theta_vals = np.linspace(0, 2 * np.pi, ntheta)
    R, THETA = np.meshgrid(r_vals, theta_vals, indexing='ij')

    # Evaluate agent output against ground truth
    print("\n--- Agent Metrics ---")
    try:
        metrics_agent = evaluate_results(gt_stress_sum, recon_stress_sum, R, THETA, delta_T_noisy)
    except Exception as e:
        print(f"ERROR in evaluate_results (agent): {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate ground truth against itself (should be perfect)
    print("\n--- Standard (GT vs GT) Metrics ---")
    try:
        metrics_std = evaluate_results(gt_stress_sum, gt_stress_sum, R, THETA, delta_T_noisy)
    except Exception as e:
        print(f"ERROR in evaluate_results (std): {e}")
        traceback.print_exc()
        sys.exit(1)

    # ─── Verification ───
    print("\n=== Verification ===")
    score_agent = metrics_agent['PSNR']
    score_std = metrics_std['PSNR']
    ssim_agent = metrics_agent['SSIM']
    cc_agent = metrics_agent['CC']

    print(f"Scores -> Agent PSNR: {score_agent:.2f} dB, Standard PSNR: {score_std:.2f} dB")
    print(f"Agent SSIM: {ssim_agent:.4f}, Agent CC: {cc_agent:.6f}")

    # Also do a direct numerical comparison
    max_abs_diff = np.max(np.abs(gt_stress_sum - recon_stress_sum))
    mean_abs_diff = np.mean(np.abs(gt_stress_sum - recon_stress_sum))
    print(f"Max absolute diff: {max_abs_diff:.6e}")
    print(f"Mean absolute diff: {mean_abs_diff:.6e}")

    # For this simple algebraic inversion, we expect very high fidelity
    # Check multiple criteria:
    # 1. PSNR should be very high (>= 40 dB for near-perfect reconstruction, or at least 90% of std)
    # 2. SSIM should be very close to 1
    # 3. CC should be very close to 1

    passed = True
    reasons = []

    # Since the function is a simple algebraic operation (-delta_T / thermo_coeff),
    # the agent output should be essentially identical to ground truth.
    # We allow a generous margin in case of minor floating point differences.

    if score_agent < 40.0 and score_agent < score_std * 0.9:
        passed = False
        reasons.append(f"PSNR too low: {score_agent:.2f} dB (std: {score_std:.2f} dB)")

    if ssim_agent < 0.9:
        passed = False
        reasons.append(f"SSIM too low: {ssim_agent:.4f}")

    if cc_agent < 0.9:
        passed = False
        reasons.append(f"CC too low: {cc_agent:.6f}")

    if passed:
        print("\n✅ TEST PASSED: Agent performance is acceptable.")
        sys.exit(0)
    else:
        print(f"\n❌ TEST FAILED: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)