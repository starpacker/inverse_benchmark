import sys
import os
import dill
import numpy as np
import traceback

# Import the target function from agent
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim_fn

# Inject the evaluate_results function (Reference B)
def evaluate_results(
    q_gt: np.ndarray,
    q_rec: np.ndarray,
    T_clean: np.ndarray,
    T_noisy: np.ndarray,
    t: np.ndarray,
    results_dir: str
) -> dict:
    """
    Compute metrics, visualize, and save results.

    Parameters
    ----------
    q_gt : np.ndarray
        Ground truth heat flux, shape (nt,).
    q_rec : np.ndarray
        Reconstructed heat flux, shape (nt,).
    T_clean : np.ndarray
        Clean sensor temperature, shape (nt,).
    T_noisy : np.ndarray
        Noisy sensor temperature, shape (nt,).
    t : np.ndarray
        Time array, shape (nt,).
    results_dir : str
        Directory to save results.

    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, CC, RE, RMSE.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Apply optimal affine alignment to correct regularisation-induced bias
    A_aff = np.vstack([q_rec, np.ones(len(q_rec))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_aff, q_gt, rcond=None)
    q_rec_aligned = coeffs[0] * q_rec + coeffs[1]
    print(f"[METRICS] Affine alignment: a={coeffs[0]:.4f}, b={coeffs[1]:.1f}")
    print(f"[METRICS] Raw PSNR before alignment: {10 * np.log10((q_gt.max() - q_gt.min()) ** 2 / max(np.mean((q_gt - q_rec) ** 2), 1e-30)):.2f} dB")

    # Use aligned reconstruction for metrics
    q_eval = q_rec_aligned

    dr = q_gt.max() - q_gt.min()
    mse = np.mean((q_gt - q_eval) ** 2)
    psnr = float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))

    tile_rows = 7
    a2d = np.tile(q_gt, (tile_rows, 1))
    b2d = np.tile(q_eval, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))

    cc = float(np.corrcoef(q_gt, q_eval)[0, 1])
    re = float(np.linalg.norm(q_gt - q_eval) / max(np.linalg.norm(q_gt), 1e-12))
    rmse = float(np.sqrt(mse))

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse
    }

    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reconstructions
    np.save(os.path.join(results_dir, "reconstruction.npy"), q_rec_aligned)
    np.save(os.path.join(results_dir, "ground_truth.npy"), q_gt)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, q_gt, 'b-', lw=2, label='GT')
    axes[0, 0].plot(t, q_rec_aligned, 'r--', lw=2, label='Recon')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Heat flux [W/m²]')
    axes[0, 0].set_title('(a) Heat Flux')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, T_clean, 'b-', lw=2, label='Clean')
    axes[0, 1].plot(t, T_noisy, 'k.', ms=1, alpha=0.3, label='Noisy')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('T [°C]')
    axes[0, 1].set_title('(b) Sensor Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, q_gt - q_rec_aligned, 'g-', lw=1)
    axes[1, 0].axhline(0, color='k', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Error [W/m²]')
    axes[1, 0].set_title(f'(c) Residual  RMSE={metrics["RMSE"]:.0f}')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(
        0.5, 0.5,
        '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
        transform=axes[1, 1].transAxes,
        ha='center',
        va='center',
        fontsize=12,
        family='monospace'
    )
    axes[1, 1].set_title('Metrics')
    axes[1, 1].axis('off')

    fig.suptitle(
        f"IHCP — Inverse Heat Conduction\nPSNR={metrics['PSNR']:.1f} dB  |  CC={metrics['CC']:.4f}",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics


# Forward operator needed for generating ground truth data
from scipy.linalg import solve

def forward_operator(
    q_flux: np.ndarray,
    nx: int,
    nt: int,
    L: float,
    t_total: float,
    alpha: float,
    k_cond: float,
    sensor_pos: float
) -> tuple:
    """
    Solve 1D heat equation with Crank-Nicolson and return temperature at sensor location.
    """
    dx = L / (nx - 1)
    dt = t_total / nt
    x = np.linspace(0, L, nx)

    r = alpha * dt / (2 * dx**2)

    T = np.zeros(nx)
    T_field = np.zeros((nx, nt))

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))

    for i in range(1, nx - 1):
        A[i, i - 1] = -r
        A[i, i] = 1 + 2 * r
        A[i, i + 1] = -r
        B[i, i - 1] = r
        B[i, i] = 1 - 2 * r
        B[i, i + 1] = r

    A[0, 0] = 1 + 2 * r
    A[0, 1] = -2 * r
    A[-1, -1] = 1
    B[0, 0] = 1 - 2 * r
    B[0, 1] = 2 * r
    B[-1, -1] = 1

    ix_sensor = int(np.argmin(np.abs(x - sensor_pos)))

    T_sensor = np.zeros(nt)

    for n in range(nt):
        rhs = B @ T
        rhs[0] += 2 * r * dx * q_flux[n] / k_cond
        T = solve(A, rhs)
        T_field[:, n] = T
        T_sensor[n] = T[ix_sensor]

    return T_sensor, T_field


def main():
    # Data paths provided
    data_paths = ['/data/yjh/IHCP_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Parse data paths to identify outer and inner data
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    print(f"[TEST] Loading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs and expected output
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[TEST] Running agent's run_inversion...")
    
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent's run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check for chained execution (inner data)
    if len(inner_data_paths) > 0:
        print(f"[TEST] Chained execution detected. Loading inner data...")
        inner_data_path = inner_data_paths[0]
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        
        # agent_output should be callable
        if callable(agent_output):
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        else:
            final_agent_result = agent_output
        
        std_result = inner_data.get('output', None)
    else:
        # Direct execution
        final_agent_result = agent_output
        std_result = std_output
    
    print("[TEST] Preparing evaluation data...")
    
    # Extract parameters from kwargs for forward operator
    # The function signature: run_inversion(T_meas, t, nx, nt, L, t_total, alpha, k_cond, sensor_x)
    # We need to reconstruct ground truth heat flux for evaluation
    
    # Extract parameters
    T_meas = args[0] if len(args) > 0 else kwargs.get('T_meas')
    t = args[1] if len(args) > 1 else kwargs.get('t')
    nx = args[2] if len(args) > 2 else kwargs.get('nx')
    nt = args[3] if len(args) > 3 else kwargs.get('nt')
    L = args[4] if len(args) > 4 else kwargs.get('L')
    t_total = args[5] if len(args) > 5 else kwargs.get('t_total')
    alpha = args[6] if len(args) > 6 else kwargs.get('alpha')
    k_cond = args[7] if len(args) > 7 else kwargs.get('k_cond')
    sensor_x = args[8] if len(args) > 8 else kwargs.get('sensor_x')
    
    # Use std_result as the ground truth heat flux (q_gt)
    q_gt = std_result
    q_agent = final_agent_result
    
    # Compute clean temperature from ground truth heat flux
    print("[TEST] Computing clean temperature from ground truth...")
    T_clean, _ = forward_operator(q_gt, nx, nt, L, t_total, alpha, k_cond, sensor_x)
    T_noisy = T_meas
    
    # Create results directory
    results_dir_agent = "./test_results_agent"
    results_dir_std = "./test_results_std"
    
    print("[TEST] Evaluating agent's reconstruction...")
    try:
        metrics_agent = evaluate_results(
            q_gt=q_gt,
            q_rec=q_agent,
            T_clean=T_clean,
            T_noisy=T_noisy,
            t=t,
            results_dir=results_dir_agent
        )
    except Exception as e:
        print(f"ERROR: Evaluation of agent result failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n[TEST] Evaluating standard reconstruction...")
    try:
        metrics_std = evaluate_results(
            q_gt=q_gt,
            q_rec=std_result,
            T_clean=T_clean,
            T_noisy=T_noisy,
            t=t,
            results_dir=results_dir_std
        )
    except Exception as e:
        print(f"ERROR: Evaluation of standard result failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metric for comparison
    # Using PSNR as the primary metric (higher is better)
    psnr_agent = metrics_agent.get('PSNR', 0)
    psnr_std = metrics_std.get('PSNR', 0)
    
    cc_agent = metrics_agent.get('CC', 0)
    cc_std = metrics_std.get('CC', 0)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS:")
    print(f"{'='*60}")
    print(f"Metric        | Agent       | Standard")
    print(f"-" * 45)
    print(f"PSNR (dB)     | {psnr_agent:11.4f} | {psnr_std:.4f}")
    print(f"CC            | {cc_agent:11.4f} | {cc_std:.4f}")
    print(f"SSIM          | {metrics_agent.get('SSIM', 0):11.4f} | {metrics_std.get('SSIM', 0):.4f}")
    print(f"RE            | {metrics_agent.get('RE', 0):11.4f} | {metrics_std.get('RE', 0):.4f}")
    print(f"RMSE          | {metrics_agent.get('RMSE', 0):11.4f} | {metrics_std.get('RMSE', 0):.4f}")
    print(f"{'='*60}")
    
    # Determine success based on PSNR and CC
    # Allow 10% margin for PSNR degradation
    # For PSNR, higher is better
    psnr_threshold = psnr_std * 0.9 if psnr_std > 0 else psnr_std * 1.1
    cc_threshold = cc_std * 0.95 if cc_std > 0 else cc_std * 1.05
    
    success = True
    
    if psnr_std > 0:
        if psnr_agent < psnr_threshold:
            print(f"[FAIL] PSNR degraded significantly: {psnr_agent:.4f} < {psnr_threshold:.4f} (90% of standard)")
            success = False
        else:
            print(f"[PASS] PSNR acceptable: {psnr_agent:.4f} >= {psnr_threshold:.4f}")
    else:
        if psnr_agent > psnr_threshold:
            print(f"[FAIL] PSNR degraded significantly: {psnr_agent:.4f} > {psnr_threshold:.4f}")
            success = False
        else:
            print(f"[PASS] PSNR acceptable: {psnr_agent:.4f}")
    
    if cc_std > 0:
        if cc_agent < cc_threshold:
            print(f"[FAIL] CC degraded significantly: {cc_agent:.4f} < {cc_threshold:.4f} (95% of standard)")
            success = False
        else:
            print(f"[PASS] CC acceptable: {cc_agent:.4f} >= {cc_threshold:.4f}")
    else:
        print(f"[PASS] CC acceptable: {cc_agent:.4f}")
    
    if success:
        print("\n[SUCCESS] Agent's run_inversion performance is acceptable.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Agent's run_inversion performance degraded significantly.")
        sys.exit(1)


if __name__ == "__main__":
    main()