import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the agent code to sys.path if necessary
# assuming agent_load_and_preprocess_data.py is in the current directory or python path
try:
    from agent_load_and_preprocess_data import load_and_preprocess_data
except ImportError:
    # If strictly running in a sandbox where the file is local
    sys.path.append(os.getcwd())
    from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is not in path (though instruction says it is)
    print("Warning: verification_utils not found. Using basic equality check.")
    def recursive_check(expected, actual):
        try:
            if isinstance(expected, np.ndarray):
                if not np.allclose(expected, actual, rtol=1e-4, atol=1e-6):
                    return False, f"Arrays differ. Max diff: {np.max(np.abs(expected - actual))}"
            elif isinstance(expected, (list, tuple)):
                if len(expected) != len(actual):
                    return False, "Length mismatch"
                for i, (e, a) in enumerate(zip(expected, actual)):
                    p, m = recursive_check(e, a)
                    if not p: return False, f"Index {i}: {m}"
            return True, "Match"
        except Exception as e:
            return False, str(e)

# Global Mock/Helper Setup for dill loading
# We need to ensure that any custom classes/functions serialized in the pickle
# are available in the global namespace so dill can reconstruct them.
# The user provided reference code contains several helpers. We will mock/define
# necessary ones or trust they are handled if they are simple numpy types.
# However, the best practice with dill and complex dependencies is to ensure imports exist.

def run_test():
    print("Starting test for load_and_preprocess_data...")
    
    # Paths provided in instructions
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify File Types
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if "standard_data_load_and_preprocess_data.pkl" in p:
            outer_path = p
        elif "parent_function_load_and_preprocess_data" in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data_load_and_preprocess_data.pkl not found.")
        sys.exit(1)

    # 2. Load Outer Data
    print(f"Loading data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. Execute Target Function
    print("Executing load_and_preprocess_data...")
    try:
        actual_output = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification Logic
    # Scenario A: The function returns data directly (tuples of arrays)
    # Scenario B: The function returns a closure/object (unlikely here based on code analysis, 
    # but strictly following the Factory pattern logic just in case).
    
    # Based on the provided code, load_and_preprocess_data returns:
    # (b_scaled, mask, Vim, im_ref)
    
    # NOTE: 'mask' and 'b_scaled' involve random sampling (mask2d uses np.random).
    # Unless seeds were fixed identically during capture and playback, 
    # the mask and undersampled data will differ.
    # Verification strategy:
    # 1. Check Shapes/Types for stochastic outputs.
    # 2. Check Values for deterministic outputs (Vim - coil maps, im_ref - reference image).
    
    print("Verifying results...")
    
    if isinstance(expected_output, tuple) and len(expected_output) == 4:
        exp_b, exp_mask, exp_Vim, exp_im_ref = expected_output
        act_b, act_mask, act_Vim, act_im_ref = actual_output
        
        # Stochastic Check (Shapes/Dtypes)
        failed_checks = []
        if exp_b.shape != act_b.shape: failed_checks.append("b_scaled shape mismatch")
        if exp_mask.shape != act_mask.shape: failed_checks.append("mask shape mismatch")
        
        if failed_checks:
            print(f"Verification Failed on Stochastic components: {failed_checks}")
            sys.exit(1)
        else:
            print("Stochastic components (b_scaled, mask) shape verification passed.")

        # Deterministic Check (Values)
        # Note: Coil sensitivity estimation (ESPIRiT) uses SVD which is generally stable 
        # but might have minor float diffs. im_ref is derived from deterministic FFT of raw data.
        
        # We group the deterministic parts for verification
        expected_deterministic = (exp_Vim, exp_im_ref)
        actual_deterministic = (act_Vim, act_im_ref)
        
        # Removed 'tol' argument as it caused TypeError in previous run
        passed, msg = recursive_check(expected_deterministic, actual_deterministic)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # If explicit failure, we might want to check if it's just minor float noise 
            # that recursive_check is too strict about, but usually recursive_check handles numpy arrays well.
            sys.exit(1)

    else:
        # Fallback for unexpected return structure
        passed, msg = recursive_check(expected_output, actual_output)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()