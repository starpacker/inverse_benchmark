import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# --- Injected Referee (Evaluation Logic) ---
def evaluate_results(F, preprocessed_data, outname):
    """
    Save results and print statistics.
    
    Parameters
    ----------
    F : ndarray
        Unwrapped phase result.
    preprocessed_data : dict
        Dictionary containing magnitude and other metadata.
    outname : str
        Output filename.
        
    Returns
    -------
    mean_phase : float
        Mean value of the unwrapped phase.
    """
    mag = preprocessed_data['mag']

    min_val = np.min(F)
    max_val = np.max(F)
    mean_val = np.mean(F)
    std_val = np.std(F)

    print(f"Evaluation: Unwrapped phase range [{min_val}, {max_val}]")
    print(f"Evaluation: Mean={mean_val}, Std={std_val}")

    if outname.endswith(".tif"):
        try:
            import rasterio as rio
            height, width = F.shape
            with rio.open(
                outname,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                dtype=F.dtype,
                count=1,
            ) as dst:
                dst.write(F, 1)
            print(f"Saved result to {outname}")
        except ImportError:
            print("rasterio not found, saving as npy instead")
            np.save(outname.replace(".tif", ".npy"), F)
            print(f"Saved numpy result to {outname.replace('.tif', '.npy')}")

    elif outname.endswith(".unw"):
        unw_with_mag = np.hstack((mag, F))
        unw_with_mag.tofile(outname)
        print(f"Saved binary result to {outname}")
    else:
        # Default fallback, just save npy
        np.save(outname + ".npy", F)
        print(f"Saved numpy result to {outname}.npy")

    return mean_val


def main():
    # Data paths provided
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Run the target function
        print("Running run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("run_inversion completed.")
        
        if is_chained:
            # Pattern 2: Chained Execution
            print("Detected chained execution pattern.")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            print("Executing inner function (operator)...")
            final_result = agent_output(*inner_args, **inner_kwargs)
            print("Inner function completed.")
        else:
            # Pattern 1: Direct Execution
            print("Detected direct execution pattern.")
            final_result = agent_output
            std_result = std_output
        
        # Get preprocessed_data for evaluation
        # It should be in args[0] based on the function signature
        preprocessed_data = args[0] if len(args) > 0 else kwargs.get('preprocessed_data', None)
        
        if preprocessed_data is None:
            print("ERROR: Could not find preprocessed_data for evaluation!")
            sys.exit(1)
        
        # Create temporary output names for evaluation
        agent_outname = "/tmp/agent_result_test"
        std_outname = "/tmp/std_result_test"
        
        # Evaluation Phase
        print("\n--- Evaluating Agent Output ---")
        score_agent = evaluate_results(final_result, preprocessed_data, agent_outname)
        
        print("\n--- Evaluating Standard Output ---")
        score_std = evaluate_results(std_result, preprocessed_data, std_outname)
        
        print(f"\nScores -> Agent: {score_agent}, Standard: {score_std}")
        
        # Additional metrics for comparison
        # Compare the actual arrays
        if final_result is not None and std_result is not None:
            # Calculate difference metrics
            diff = np.abs(final_result - std_result)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # Calculate relative error
            std_range = np.max(std_result) - np.min(std_result)
            if std_range > 0:
                relative_error = max_diff / std_range
            else:
                relative_error = max_diff
            
            print(f"\nArray Comparison:")
            print(f"  Max absolute difference: {max_diff}")
            print(f"  Mean absolute difference: {mean_diff}")
            print(f"  Relative error (max_diff/range): {relative_error}")
            
            # Check correlation
            correlation = np.corrcoef(final_result.ravel(), std_result.ravel())[0, 1]
            print(f"  Correlation coefficient: {correlation}")
        
        # Verification
        # For mean_val metric, we check if they are close
        # Allow 10% margin of error for the mean value
        if score_std != 0:
            relative_diff = abs(score_agent - score_std) / abs(score_std)
        else:
            relative_diff = abs(score_agent - score_std)
        
        print(f"\nRelative difference in mean: {relative_diff:.4f}")
        
        # Success criteria:
        # 1. Mean values should be within 10% of each other
        # 2. Correlation should be high (> 0.95)
        # 3. Relative error should be small (< 0.1)
        
        success = True
        
        if relative_diff > 0.1:
            print(f"WARNING: Mean value difference ({relative_diff:.4f}) exceeds 10% threshold")
            success = False
        
        if 'correlation' in dir() and correlation < 0.95:
            print(f"WARNING: Correlation ({correlation:.4f}) is below 0.95 threshold")
            success = False
        
        if 'relative_error' in dir() and relative_error > 0.1:
            print(f"WARNING: Relative error ({relative_error:.4f}) exceeds 10% threshold")
            success = False
        
        if success:
            print("\n=== TEST PASSED ===")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Exception occurred during testing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()