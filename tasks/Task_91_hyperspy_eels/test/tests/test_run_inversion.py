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

# Inject the referee evaluation function
def evaluate_results(ground_truth, reconstruction, measured, zlp, energy_axis, results_dir):
    """
    Compute quality metrics and generate visualizations.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground truth SSD.
    reconstruction : ndarray
        Reconstructed SSD.
    measured : ndarray
        Measured EELS spectrum.
    zlp : ndarray
        Zero-loss peak.
    energy_axis : ndarray
        Energy axis in eV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, RMSE, CC, relative_error.
    """
    # Define region of interest (ROI: 2–80 eV)
    roi = (energy_axis >= 2.0) & (energy_axis <= 80.0)
    gt_roi = ground_truth[roi].astype(np.float64)
    rec_roi = reconstruction[roi].astype(np.float64)
    
    # Compute MSE and RMSE
    mse = np.mean((gt_roi - rec_roi)**2)
    rmse = np.sqrt(mse)
    
    # Compute PSNR
    data_range = np.max(gt_roi) - np.min(gt_roi)
    if data_range > 0 and rmse > 0:
        psnr = 20.0 * np.log10(data_range / rmse)
    else:
        psnr = float('inf')
    
    # Compute correlation coefficient
    gt_c = gt_roi - np.mean(gt_roi)
    rec_c = rec_roi - np.mean(rec_roi)
    denom = np.sqrt(np.sum(gt_c**2) * np.sum(rec_c**2))
    cc = float(np.sum(gt_c * rec_c) / denom) if denom > 0 else 0.0
    
    # Compute relative error
    gt_norm = np.linalg.norm(gt_roi)
    rel_err = float(np.linalg.norm(gt_roi - rec_roi) / gt_norm) if gt_norm > 0 else float('inf')
    
    metrics = {
        "PSNR": float(np.round(psnr, 4)),
        "RMSE": float(np.round(rmse, 6)),
        "CC": float(np.round(cc, 6)),
        "relative_error": float(np.round(rel_err, 6))
    }
    
    # Save metrics to JSON
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save numpy arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), ground_truth)
    np.save(os.path.join(results_dir, "reconstruction.npy"), reconstruction)
    np.save(os.path.join(results_dir, "input_measurement.npy"), measured)
    np.save(os.path.join(results_dir, "energy_axis.npy"), energy_axis)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             gridspec_kw={'height_ratios': [1.2, 1.2, 0.8]})
    xl = [0, 80]
    
    # Panel 1: Measured EELS
    ax = axes[0]
    ax.plot(energy_axis, measured, 'b-', lw=1.0, alpha=0.8,
            label='Measured EELS (multiple scattering)')
    scale = 0.3 * np.max(measured) / (np.max(zlp) + 1e-30)
    ax.plot(energy_axis, zlp * scale, 'g--', lw=1.0, alpha=0.7,
            label='Zero-Loss Peak (scaled)')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('EELS Measurement with Multiple Scattering',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: GT vs Reconstructed SSD
    ax = axes[1]
    ax.plot(energy_axis, ground_truth, 'k-', lw=2.0, label='Ground Truth SSD')
    ax.plot(energy_axis, reconstruction, 'r--', lw=1.5, alpha=0.9,
            label='Reconstructed SSD (Fourier-Log)')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title(
        f'Single Scattering Distribution Recovery\n'
        f'PSNR = {metrics["PSNR"]:.2f} dB | CC = {metrics["CC"]:.4f} | '
        f'RMSE = {metrics["RMSE"]:.4e}',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Residual
    ax = axes[2]
    residual = ground_truth - reconstruction
    ax.plot(energy_axis, residual, 'purple', lw=1.0, alpha=0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.fill_between(energy_axis, residual, alpha=0.2, color='purple')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Residual (GT − Reconstruction)', fontsize=13, fontweight='bold')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/hyperspy_eels_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Results directory
    results_dir = './test_results_run_inversion'
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args length: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Run the agent's function
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_files:
            print("Chained execution detected - loading inner data...")
            inner_path = inner_files[0]
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned callable
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        print(f"Agent output type: {type(final_result)}")
        print(f"Standard output type: {type(std_result)}")
        
        # Extract results from dictionaries
        if isinstance(final_result, dict):
            agent_ssd = final_result.get('ssd_recovered', None)
            agent_energy = final_result.get('energy_axis', None)
        else:
            agent_ssd = final_result
            agent_energy = None
        
        if isinstance(std_result, dict):
            std_ssd = std_result.get('ssd_recovered', None)
            std_energy = std_result.get('energy_axis', None)
        else:
            std_ssd = std_result
            std_energy = None
        
        # Get input parameters for evaluation
        measured_spectrum = args[0] if len(args) > 0 else kwargs.get('measured_spectrum')
        zlp = args[1] if len(args) > 1 else kwargs.get('zlp')
        energy_axis = args[2] if len(args) > 2 else kwargs.get('energy_axis')
        
        if energy_axis is None and agent_energy is not None:
            energy_axis = agent_energy
        
        print(f"Agent SSD shape: {agent_ssd.shape if agent_ssd is not None else None}")
        print(f"Standard SSD shape: {std_ssd.shape if std_ssd is not None else None}")
        print(f"Energy axis shape: {energy_axis.shape if energy_axis is not None else None}")
        
        # Use standard result as ground truth for evaluation
        # The standard result represents the expected output
        ground_truth = std_ssd
        reconstruction_agent = agent_ssd
        
        # Evaluate agent's results
        print("\nEvaluating Agent's output...")
        agent_results_dir = os.path.join(results_dir, 'agent')
        metrics_agent = evaluate_results(
            ground_truth=ground_truth,
            reconstruction=reconstruction_agent,
            measured=measured_spectrum,
            zlp=zlp,
            energy_axis=energy_axis,
            results_dir=agent_results_dir
        )
        
        # Evaluate standard results (should be perfect match with itself)
        print("Evaluating Standard output...")
        std_results_dir = os.path.join(results_dir, 'standard')
        metrics_std = evaluate_results(
            ground_truth=ground_truth,
            reconstruction=std_ssd,
            measured=measured_spectrum,
            zlp=zlp,
            energy_axis=energy_axis,
            results_dir=std_results_dir
        )
        
        # Extract primary metrics
        psnr_agent = metrics_agent['PSNR']
        psnr_std = metrics_std['PSNR']
        cc_agent = metrics_agent['CC']
        cc_std = metrics_std['CC']
        rmse_agent = metrics_agent['RMSE']
        rmse_std = metrics_std['RMSE']
        rel_err_agent = metrics_agent['relative_error']
        rel_err_std = metrics_std['relative_error']
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Agent Metrics:")
        print(f"  PSNR: {psnr_agent:.4f} dB")
        print(f"  CC: {cc_agent:.6f}")
        print(f"  RMSE: {rmse_agent:.6e}")
        print(f"  Relative Error: {rel_err_agent:.6f}")
        print(f"\nStandard Metrics:")
        print(f"  PSNR: {psnr_std:.4f} dB")
        print(f"  CC: {cc_std:.6f}")
        print(f"  RMSE: {rmse_std:.6e}")
        print(f"  Relative Error: {rel_err_std:.6f}")
        print(f"{'='*60}")
        
        # Determine success based on metrics
        # Since we're comparing agent output to standard output (which should be identical
        # if the implementation is correct), we expect perfect or near-perfect scores
        
        # For PSNR: higher is better (should be very high or inf for identical results)
        # For CC: higher is better (should be 1.0 for identical results)
        # For RMSE: lower is better (should be 0 for identical results)
        # For relative_error: lower is better (should be 0 for identical results)
        
        success = True
        tolerance = 0.1  # 10% tolerance
        
        # Check if results are essentially identical
        if not np.isinf(psnr_agent):
            if psnr_agent < 40:  # PSNR below 40 dB indicates significant difference
                print(f"WARNING: PSNR is below 40 dB, indicating potential issues")
                if psnr_agent < 20:  # Very low PSNR
                    success = False
                    print("FAIL: PSNR too low")
        
        if cc_agent < 0.99:  # Correlation should be very high
            print(f"WARNING: Correlation coefficient is below 0.99")
            if cc_agent < 0.9:
                success = False
                print("FAIL: Correlation too low")
        
        if rel_err_agent > 0.01:  # Relative error should be very small
            print(f"WARNING: Relative error is above 1%")
            if rel_err_agent > 0.1:
                success = False
                print("FAIL: Relative error too high")
        
        # Direct array comparison as additional check
        array_diff = np.max(np.abs(agent_ssd - std_ssd))
        print(f"\nMax absolute difference between arrays: {array_diff:.6e}")
        
        if array_diff < 1e-10:
            print("Arrays are essentially identical (difference < 1e-10)")
        elif array_diff < 1e-6:
            print("Arrays are very close (difference < 1e-6)")
        else:
            print(f"Arrays have measurable differences")
        
        if success:
            print(f"\n{'='*60}")
            print("TEST PASSED: Agent performance is acceptable")
            print(f"{'='*60}")
            sys.exit(0)
        else:
            print(f"\n{'='*60}")
            print("TEST FAILED: Agent performance degraded significantly")
            print(f"{'='*60}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Test execution failed!")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()