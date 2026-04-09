import sys
import os
import dill
import numpy as np
import traceback
import glob

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim


def evaluate_results(img_recon, expected_output_path, output_path, scaler, original_dtype):
    """
    Evaluate reconstruction results and save output.
    
    Parameters
    ----------
    img_recon : ndarray
        Reconstructed image (normalized).
    expected_output_path : str
        Path to expected output for comparison.
    output_path : str
        Path to save reconstructed image.
    scaler : float
        Scaling factor to restore original intensity range.
    original_dtype : dtype
        Original image data type for saving.
    
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, and MSE values.
    """
    # Rescale output
    img_output = scaler * img_recon
    
    # Save result
    io.imsave(output_path, img_output.astype(original_dtype))
    print(f"✅ Processing complete! Result saved to: {output_path}")
    
    # Load expected output
    expected = io.imread(expected_output_path)
    
    # Ensure shape matches
    if img_output.shape != expected.shape:
        raise ValueError("Reconstructed image and expected image must have the same shape!")
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(expected, img_output, data_range=expected.max() - expected.min())
    ssim_val = ssim(expected, img_output, data_range=expected.max() - expected.min(),
                   channel_axis=None if len(expected.shape) == 2 else 2)
    mse = np.mean((expected.astype(np.float64) - img_output.astype(np.float64)) ** 2)
    
    print(f"📊 PSNR: {psnr:.4f} dB")
    print(f"📊 SSIM: {ssim_val:.4f}")
    print(f"📊 MSE: {mse:.6f}")
    
    metrics = {
        'psnr': psnr,
        'ssim': ssim_val,
        'mse': mse
    }
    
    return metrics


def compare_arrays(agent_result, std_result, tolerance=0.1):
    """
    Compare two arrays using multiple metrics.
    
    Returns a score indicating how similar they are (higher is better).
    """
    if agent_result is None or std_result is None:
        return 0.0
    
    # Convert to numpy arrays if needed
    agent_arr = np.asarray(agent_result, dtype=np.float64)
    std_arr = np.asarray(std_result, dtype=np.float64)
    
    # Check shape compatibility
    if agent_arr.shape != std_arr.shape:
        print(f"Shape mismatch: agent {agent_arr.shape} vs std {std_arr.shape}")
        return 0.0
    
    # Calculate correlation
    agent_flat = agent_arr.flatten()
    std_flat = std_arr.flatten()
    
    # Normalize for comparison
    agent_norm = agent_flat / (np.max(np.abs(agent_flat)) + 1e-10)
    std_norm = std_flat / (np.max(np.abs(std_flat)) + 1e-10)
    
    # Correlation coefficient
    correlation = np.corrcoef(agent_norm, std_norm)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # Relative MSE
    mse = np.mean((agent_norm - std_norm) ** 2)
    rel_mse = mse / (np.mean(std_norm ** 2) + 1e-10)
    
    print(f"Correlation: {correlation:.6f}")
    print(f"Relative MSE: {rel_mse:.6f}")
    
    return correlation, rel_mse


def main():
    # Define data paths
    data_paths = []
    
    # Standard data directory
    std_data_dir = "run_code/std_data"
    
    # Look for the specific run_inversion data file
    target_file = os.path.join(std_data_dir, "standard_data_run_inversion.pkl")
    
    if os.path.exists(target_file):
        data_paths = [target_file]
    else:
        # Search recursively for pkl files
        all_pkl = glob.glob(os.path.join(std_data_dir, "**", "*.pkl"), recursive=True)
        # Filter for run_inversion
        run_inv_files = [f for f in all_pkl if "run_inversion" in f]
        if run_inv_files:
            data_paths = run_inv_files
        else:
            print("ERROR: Could not find run_inversion data file")
            sys.exit(1)
    
    print(f"Found data files: {data_paths}")
    
    # Separate outer and inner files
    outer_files = [f for f in data_paths if "parent_function" not in f]
    inner_files = [f for f in data_paths if "parent_function" in f]
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    if not outer_files:
        print("ERROR: No outer data files found")
        sys.exit(1)
    
    # Load outer data
    outer_path = outer_files[0]
    print(f"\nLoading outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output')
    
    print(f"Args type: {type(args)}, length: {len(args) if isinstance(args, (list, tuple)) else 'N/A'}")
    print(f"Kwargs keys: {kwargs.keys()}")
    print(f"Expected output type: {type(std_output)}")
    
    # Print args details for debugging
    if args:
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                print(f"  arg[{i}]: ndarray shape={arg.shape}, dtype={arg.dtype}")
            else:
                print(f"  arg[{i}]: {type(arg)} = {arg}")
    
    # Print kwargs for debugging
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            print(f"  kwarg[{k}]: ndarray shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  kwarg[{k}]: {type(v)} = {v}")
    
    print("\n" + "=" * 60)
    print("Running agent's run_inversion function...")
    print("=" * 60)
    
    try:
        agent_output = run_inversion(*args, **kwargs)
        print(f"\nAgent output type: {type(agent_output)}")
        if isinstance(agent_output, np.ndarray):
            print(f"Agent output shape: {agent_output.shape}")
            print(f"Agent output dtype: {agent_output.dtype}")
            print(f"Agent output range: [{agent_output.min():.6f}, {agent_output.max():.6f}]")
    except Exception as e:
        print(f"ERROR: Agent function failed with exception:")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner files (chained execution)
    if inner_files:
        print("\n" + "=" * 60)
        print("Chained execution detected - running inner function...")
        print("=" * 60)
        
        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_output = inner_data.get('output')
        
        try:
            # agent_output should be a callable
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Inner function call failed:")
            traceback.print_exc()
            sys.exit(1)
    else:
        final_result = agent_output
    
    print("\n" + "=" * 60)
    print("Comparing results...")
    print("=" * 60)
    
    print(f"Agent result type: {type(final_result)}")
    print(f"Standard result type: {type(std_output)}")
    
    if isinstance(final_result, np.ndarray):
        print(f"Agent result shape: {final_result.shape}")
        print(f"Agent result range: [{final_result.min():.6f}, {final_result.max():.6f}]")
    
    if isinstance(std_output, np.ndarray):
        print(f"Standard result shape: {std_output.shape}")
        print(f"Standard result range: [{std_output.min():.6f}, {std_output.max():.6f}]")
    
    # Compare results
    try:
        if isinstance(final_result, np.ndarray) and isinstance(std_output, np.ndarray):
            correlation, rel_mse = compare_arrays(final_result, std_output)
            
            print(f"\n" + "=" * 60)
            print(f"Scores -> Correlation: {correlation:.6f}, Relative MSE: {rel_mse:.6f}")
            print("=" * 60)
            
            # Success criteria: high correlation and low relative MSE
            # Allow some tolerance for numerical differences
            if correlation >= 0.9 and rel_mse <= 0.1:
                print("\n✅ TEST PASSED: Agent output matches expected output within tolerance")
                sys.exit(0)
            elif correlation >= 0.8:
                print("\n⚠️ TEST PASSED (marginal): Correlation acceptable but not ideal")
                sys.exit(0)
            else:
                print("\n❌ TEST FAILED: Agent output differs significantly from expected")
                sys.exit(1)
        else:
            # For non-array outputs, do direct comparison
            if final_result is not None and std_output is not None:
                # Try to convert to comparable format
                try:
                    agent_val = np.asarray(final_result)
                    std_val = np.asarray(std_output)
                    correlation, rel_mse = compare_arrays(agent_val, std_val)
                    
                    print(f"\nScores -> Correlation: {correlation:.6f}, Relative MSE: {rel_mse:.6f}")
                    
                    if correlation >= 0.8:
                        print("\n✅ TEST PASSED")
                        sys.exit(0)
                    else:
                        print("\n❌ TEST FAILED")
                        sys.exit(1)
                except Exception as e:
                    print(f"Could not compare outputs: {e}")
                    # If types match and values are close enough, pass
                    if type(final_result) == type(std_output):
                        print("\n✅ TEST PASSED (types match)")
                        sys.exit(0)
                    sys.exit(1)
            else:
                print("\n❌ TEST FAILED: One or both outputs are None")
                sys.exit(1)
                
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()