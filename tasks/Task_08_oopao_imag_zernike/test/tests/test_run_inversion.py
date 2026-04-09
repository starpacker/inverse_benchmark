import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# --- Injected Referee (Evaluation Logic) ---

def forward_operator(phase_map, tel):
    """
    Computes the PSF from the phase map using physical optics principles.
    PSF = | FFT( Amplitude * exp(i * Phase) ) |^2
    
    Args:
        phase_map: 2D array of phase values [radians]
        tel: Telescope object containing pupil information
        
    Returns:
        psf: 2D array of the Point Spread Function (normalized)
    """
    # 1. Get Pupil Amplitude (Binary mask)
    amplitude = tel.pupil
    
    # 2. Create Complex Field (Electric Field)
    # E = A * e^(i * phi)
    electric_field = amplitude * np.exp(1j * phase_map)
    
    # 3. Apply Zero Padding (for sampling)
    zero_padding = 4
    N = tel.resolution
    N_padded = N * zero_padding
    
    # Pad the electric field
    pad_width = (N_padded - N) // 2
    electric_field_padded = np.pad(electric_field, pad_width)
    
    # 4. Fourier Transform (Propagation to Focal Plane)
    # Shift before FFT to center zero frequency
    complex_focal_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(electric_field_padded)))
    
    # 5. Compute Intensity (PSF)
    psf = np.abs(complex_focal_plane)**2
    
    # Normalize
    psf = psf / psf.max()
    
    return psf

def evaluate_results(data_dict, inversion_results, output_dir='.'):
    """
    Evaluates and visualizes the results of forward modeling and inversion.
    
    Args:
        data_dict: Dictionary containing input data and forward model results
        inversion_results: Dictionary containing inversion results
        output_dir: Directory to save output figures
    """
    print("\n[5] Evaluating Results...")
    
    tel = data_dict['telescope']
    phase_map = data_dict['phase_map']
    rmse_history = inversion_results['rmse_history']
    
    # Compute PSF using forward operator
    print("    Computing PSF via FFT...")
    psf = forward_operator(phase_map, tel)
    
    # Plot forward model results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(phase_map * tel.pupil)
    plt.title("Input Phase (Explicit Zernikes)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(psf + 1e-10))
    plt.title("Resulting PSF (Log)")
    plt.colorbar()
    forward_plot_path = os.path.join(output_dir, "zernike_forward.png")
    plt.savefig(forward_plot_path)
    plt.close()
    print(f"    Saved {forward_plot_path}")
    
    # Plot inversion results
    plt.figure()
    plt.plot(rmse_history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE [m]')
    plt.title('Zernike Decomposition Residual')
    inverse_plot_path = os.path.join(output_dir, "zernike_inverse.png")
    plt.savefig(inverse_plot_path)
    plt.close()
    print(f"    Saved {inverse_plot_path}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"    Mean RMSE: {inversion_results['mean_rmse'] * 1e9:.2f} nm")
    print(f"    Final RMSE: {inversion_results['final_rmse'] * 1e9:.2f} nm")
    print(f"    Min RMSE: {np.min(rmse_history) * 1e9:.2f} nm")
    print(f"    Max RMSE: {np.max(rmse_history) * 1e9:.2f} nm")
    print(f"    Std RMSE: {np.std(rmse_history) * 1e9:.2f} nm")
    
    # Compare original vs reconstructed OPD for last iteration
    if len(inversion_results['all_opd_original']) > 0:
        last_original = inversion_results['all_opd_original'][-1]
        last_reconstructed = inversion_results['all_opd_reconstructed'][-1]
        
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(last_original * tel.pupil)
        plt.title("Original OPD")
        plt.colorbar(label='[m]')
        
        plt.subplot(1, 3, 2)
        plt.imshow(last_reconstructed * tel.pupil)
        plt.title("Reconstructed OPD")
        plt.colorbar(label='[m]')
        
        plt.subplot(1, 3, 3)
        residual = (last_original - last_reconstructed) * tel.pupil
        plt.imshow(residual)
        plt.title("Residual")
        plt.colorbar(label='[m]')
        
        comparison_path = os.path.join(output_dir, "opd_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"    Saved {comparison_path}")
    
    return {
        'psf': psf,
        'forward_plot_path': forward_plot_path,
        'inverse_plot_path': inverse_plot_path,
    }


def compute_performance_metric(inversion_results):
    """
    Extract a scalar performance metric from inversion results.
    Lower RMSE is better, so we return the mean RMSE as primary metric.
    """
    return inversion_results['mean_rmse']


def main():
    # Data paths provided
    data_paths = ['/home/yjh/oopao_zernike_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 60)
    print("Testing run_inversion Performance")
    print("=" * 60)
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"\nOuter data files: {outer_files}")
    print(f"Inner data files: {inner_files}")
    
    # Determine execution pattern
    is_chained = len(inner_files) > 0
    print(f"Execution pattern: {'Chained' if is_chained else 'Direct'}")
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"\nLoading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args keys/types: {[type(a).__name__ for a in args]}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent function
        print("\n" + "-" * 40)
        print("Running agent's run_inversion...")
        print("-" * 40)
        
        agent_output = run_inversion(*args, **kwargs)
        
        print("\nAgent output type:", type(agent_output).__name__)
        if isinstance(agent_output, dict):
            print("Agent output keys:", list(agent_output.keys()))
        
        if is_chained:
            # Chained execution - agent_output is a callable
            inner_path = inner_files[0]
            print(f"\nLoading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print("Executing chained call with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        print("\n" + "-" * 40)
        print("Evaluation Phase")
        print("-" * 40)
        
        # Extract the data_dict needed for evaluate_results
        # The input to run_inversion is a data_dict
        if len(args) > 0:
            data_dict = args[0]
        else:
            data_dict = kwargs.get('data_dict', {})
        
        # Create output directory for plots
        output_dir = './test_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute performance metrics
        print("\nComputing agent performance metric...")
        agent_metric = compute_performance_metric(final_result)
        print(f"Agent Mean RMSE: {agent_metric * 1e9:.4f} nm")
        
        print("\nComputing standard performance metric...")
        std_metric = compute_performance_metric(std_result)
        print(f"Standard Mean RMSE: {std_metric * 1e9:.4f} nm")
        
        # Run full evaluation for agent (generates plots)
        print("\nRunning full evaluation for agent output...")
        try:
            eval_result = evaluate_results(data_dict, final_result, output_dir)
            print("Evaluation completed successfully.")
        except Exception as e:
            print(f"Warning: Full evaluation failed: {e}")
            traceback.print_exc()
        
        # Performance comparison
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"Agent Mean RMSE:    {agent_metric * 1e9:.4f} nm")
        print(f"Standard Mean RMSE: {std_metric * 1e9:.4f} nm")
        
        # For RMSE, lower is better
        # Allow 10% margin: agent should not be more than 10% worse
        threshold = 1.10  # 10% tolerance
        
        if std_metric > 0:
            ratio = agent_metric / std_metric
            print(f"Ratio (Agent/Standard): {ratio:.4f}")
            
            if ratio <= threshold:
                print(f"\n✓ PASS: Agent performance is acceptable (ratio <= {threshold})")
                sys.exit(0)
            else:
                print(f"\n✗ FAIL: Agent performance degraded significantly (ratio > {threshold})")
                sys.exit(1)
        else:
            # Edge case: standard metric is zero
            if agent_metric <= 1e-12:  # Both effectively zero
                print("\n✓ PASS: Both metrics effectively zero")
                sys.exit(0)
            else:
                print(f"\n✗ FAIL: Standard RMSE is zero but agent RMSE is {agent_metric}")
                sys.exit(1)
                
    except Exception as e:
        print(f"\n✗ ERROR: Test failed with exception:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()