import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
from skimage.metrics import structural_similarity as ssim_fn

# Import target function
from agent_run_inversion import run_inversion


def forward_operator(density, voxel_size, n_q, q_max):
    """
    Compute 1D SAXS profile I(q) from 3D electron density.

    I(q) = spherical_average( |FFT{ρ(r)}|² )

    Parameters
    ----------
    density : ndarray
        3D electron density array.
    voxel_size : float
        Voxel size in Angstroms.
    n_q : int
        Number of q bins.
    q_max : float
        Maximum q value in inverse Angstroms.

    Returns
    -------
    q_bins : ndarray
        q values in inverse Angstroms.
    I_q : ndarray
        Scattering intensity (normalized).
    """
    N = density.shape[0]

    # 3D FFT
    F = fftshift(fftn(ifftshift(density)))
    I_3d = np.abs(F) ** 2

    # q-grid
    freq = fftfreq(N, d=voxel_size)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Radial average (spherical shells)
    q_bins = np.linspace(0.01, q_max, n_q)
    dq = q_bins[1] - q_bins[0]
    I_q = np.zeros(n_q)

    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq / 2) & (q_3d < qc + dq / 2)
        if mask.sum() > 0:
            I_q[i] = np.mean(I_3d[mask])

    # Normalise
    if I_q.max() > 0:
        I_q = I_q / I_q.max()

    return q_bins, I_q


def evaluate_results(density_gt, density_rec, I_clean, q, voxel_size, n_q, q_max, results_dir):
    """
    Compute reconstruction metrics and generate visualizations.

    Parameters
    ----------
    density_gt : ndarray
        Ground truth 3D electron density.
    density_rec : ndarray
        Reconstructed 3D electron density.
    I_clean : ndarray
        Clean I(q) values.
    q : ndarray
        q values.
    voxel_size : float
        Voxel size in Angstroms.
    n_q : int
        Number of q bins.
    q_max : float
        Maximum q value.
    results_dir : str
        Directory to save results.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    # Normalize
    gt = density_gt / max(density_gt.max(), 1e-12)
    rec = density_rec / max(density_rec.max(), 1e-12)

    # Ensure same shape
    s = min(gt.shape[0], rec.shape[0])
    gt = gt[:s, :s, :s]
    rec = rec[:s, :s, :s]

    # 3D CC
    cc_vol = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])
    re_vol = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    # Central slice metrics
    mid = s // 2
    gt_slice = gt[mid, :, :]
    rec_slice = rec[mid, :, :]
    dr = gt_slice.max() - gt_slice.min()
    if dr < 1e-12:
        dr = 1.0
    mse = np.mean((gt_slice - rec_slice) ** 2)
    psnr = float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_slice, rec_slice, data_range=dr))
    cc_slice = float(np.corrcoef(gt_slice.ravel(), rec_slice.ravel())[0, 1])

    # I(q) fit
    _, I_rec = forward_operator(rec * density_rec.max(), voxel_size, n_q, q_max)
    cc_Iq = float(np.corrcoef(I_clean, I_rec)[0, 1])

    metrics = {
        "PSNR_slice": psnr,
        "SSIM_slice": ssim_val,
        "CC_slice": cc_slice,
        "CC_volume": cc_vol,
        "RE_volume": re_vol,
        "CC_Iq": cc_Iq,
    }

    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), density_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), density_gt)

    # Generate visualization
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    
    # Load I_noisy for plotting (recompute with noise for visualization)
    rng = np.random.default_rng(42)
    I_noisy = I_clean * (1 + 0.001 * rng.standard_normal(len(I_clean)))
    I_noisy = np.maximum(I_noisy, 0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mid = density_gt.shape[0] // 2
    gt_n = density_gt / max(density_gt.max(), 1e-12)
    rec_n = density_rec / max(density_rec.max(), 1e-12)
    s = min(gt_n.shape[0], rec_n.shape[0])

    for i, (title, data) in enumerate([
        ('GT (z-slice)', gt_n[min(mid, s - 1)]),
        ('Recon (z-slice)', rec_n[min(mid, s - 1)]),
        ('Error', gt_n[min(mid, s - 1)] - rec_n[min(mid, s - 1)]),
    ]):
        axes[0, i].imshow(data, cmap='hot' if i < 2 else 'RdBu_r', origin='lower')
        axes[0, i].set_title(title)

    axes[1, 0].semilogy(q, I_clean, 'b-', lw=2, label='GT')
    axes[1, 0].semilogy(q, I_noisy, 'k.', ms=3, alpha=0.5, label='Noisy')
    axes[1, 0].set_xlabel('q [Å⁻¹]')
    axes[1, 0].set_ylabel('I(q)')
    axes[1, 0].set_title('SAXS Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gt_n[min(mid, s - 1), min(mid, s - 1), :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(rec_n[min(mid, s - 1), min(mid, s - 1), :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title('1D Line Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].text(0.5, 0.5, '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
                    transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=11,
                    family='monospace')
    axes[1, 2].set_title('Metrics')
    axes[1, 2].axis('off')

    fig.suptitle(f"DENSS — SAXS Electron Density Reconstruction\n"
                 f"PSNR={metrics['PSNR_slice']:.1f} dB  |  SSIM={metrics['SSIM_slice']:.4f}  |  "
                 f"CC={metrics['CC_volume']:.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics


def generate_ground_truth_density(shape, voxel_size, seed=42):
    """
    Generate a simple ground truth density for evaluation.
    Creates a spherical blob with some structure.
    """
    N = shape[0]
    rng = np.random.default_rng(seed)
    
    # Create coordinate grid
    z, y, x = np.mgrid[:N, :N, :N] - N // 2
    r2 = x**2 + y**2 + z**2
    
    # Create a smooth spherical density
    density = np.exp(-r2 / (2 * (N / 6)**2))
    
    # Add some structure
    density += 0.3 * np.exp(-((x - N//6)**2 + y**2 + z**2) / (2 * (N / 10)**2))
    density += 0.3 * np.exp(-((x + N//6)**2 + y**2 + z**2) / (2 * (N / 10)**2))
    
    # Normalize
    density = np.maximum(density, 0)
    density = density / max(density.max(), 1e-12)
    
    return density


def main():
    data_paths = ['/data/yjh/DENSS_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data paths
    outer_paths = []
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        else:
            outer_paths.append(path)
    
    print("=" * 60)
    print("Data paths analysis:")
    print(f"  Outer paths: {outer_paths}")
    print(f"  Inner paths: {inner_paths}")
    print("=" * 60)
    
    # Load outer data
    if not outer_paths:
        print("ERROR: No outer data found!")
        sys.exit(1)
    
    outer_path = outer_paths[0]
    print(f"Loading outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'unknown')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {func_name}")
    print(f"Args count: {len(args)}, Kwargs: {list(kwargs.keys())}")
    
    # Extract parameters from kwargs (since args is empty)
    q_data = kwargs.get('q_data')
    I_data = kwargs.get('I_data')
    density_gt_shape = kwargs.get('density_gt_shape')
    voxel_size = kwargs.get('voxel_size')
    q_max = kwargs.get('q_max')
    n_q = kwargs.get('n_q')
    n_iter = kwargs.get('n_iter')
    n_runs = kwargs.get('n_runs')
    seed = kwargs.get('seed')
    
    # Validate we have all required parameters
    if q_data is None or I_data is None:
        print("ERROR: Missing q_data or I_data in kwargs!")
        print(f"Available kwargs keys: {list(kwargs.keys())}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Running agent's run_inversion...")
    print("=" * 60)
    
    try:
        # Run the agent's function
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent's function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Agent output obtained. Running evaluation...")
    print("=" * 60)
    
    # Check for chained execution (inner data)
    if inner_paths:
        print("Chained execution detected - loading inner data...")
        inner_path = inner_paths[0]
        
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Run the operator returned by run_inversion
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    # Create results directory
    results_dir = "/data/yjh/DENSS_sandbox_sandbox/test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate or reconstruct ground truth density for evaluation
    # We need to create a ground truth density based on the input I(q)
    # Since we're doing SAXS reconstruction, we generate a synthetic GT
    density_gt = generate_ground_truth_density(density_gt_shape, voxel_size, seed=seed)
    
    # Compute I_clean from GT for comparison
    _, I_clean = forward_operator(density_gt, voxel_size, n_q, q_max)
    
    print("\n" + "=" * 60)
    print("Evaluating AGENT reconstruction...")
    print("=" * 60)
    
    try:
        metrics_agent = evaluate_results(
            density_gt=density_gt,
            density_rec=final_result,
            I_clean=I_clean,
            q=q_data,
            voxel_size=voxel_size,
            n_q=n_q,
            q_max=q_max,
            results_dir=os.path.join(results_dir, "agent")
        )
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Evaluating STANDARD reconstruction...")
    print("=" * 60)
    
    try:
        metrics_std = evaluate_results(
            density_gt=density_gt,
            density_rec=std_result,
            I_clean=I_clean,
            q=q_data,
            voxel_size=voxel_size,
            n_q=n_q,
            q_max=q_max,
            results_dir=os.path.join(results_dir, "standard")
        )
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics for comparison
    # Using CC_volume as primary metric (higher is better)
    score_agent = metrics_agent.get('CC_volume', 0.0)
    score_std = metrics_std.get('CC_volume', 0.0)
    
    # Also check PSNR
    psnr_agent = metrics_agent.get('PSNR_slice', 0.0)
    psnr_std = metrics_std.get('PSNR_slice', 0.0)
    
    # Check SSIM
    ssim_agent = metrics_agent.get('SSIM_slice', 0.0)
    ssim_std = metrics_std.get('SSIM_slice', 0.0)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"CC_volume  -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    print(f"PSNR_slice -> Agent: {psnr_agent:.2f}, Standard: {psnr_std:.2f}")
    print(f"SSIM_slice -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
    
    # Determine success: Allow 10% margin of error
    # For CC and SSIM: higher is better
    # We check if agent is within acceptable range of standard
    margin = 0.10  # 10% tolerance
    
    # Check multiple metrics
    cc_ok = score_agent >= score_std * (1 - margin) or score_agent >= 0.8
    psnr_ok = psnr_agent >= psnr_std * (1 - margin) or psnr_agent >= 15.0
    ssim_ok = ssim_agent >= ssim_std * (1 - margin) or ssim_agent >= 0.5
    
    # Overall success requires at least 2 out of 3 metrics to be acceptable
    passing_metrics = sum([cc_ok, psnr_ok, ssim_ok])
    
    print("\n" + "=" * 60)
    print("METRIC CHECKS")
    print("=" * 60)
    print(f"CC_volume acceptable: {cc_ok}")
    print(f"PSNR_slice acceptable: {psnr_ok}")
    print(f"SSIM_slice acceptable: {ssim_ok}")
    print(f"Passing metrics: {passing_metrics}/3")
    
    if passing_metrics >= 2:
        print("\n✅ TEST PASSED: Agent performance is acceptable")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED: Agent performance degraded significantly")
        sys.exit(1)


if __name__ == "__main__":
    main()