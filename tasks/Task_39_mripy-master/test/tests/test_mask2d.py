import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the target module
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_mask2d import mask2d
except ImportError:
    print("Error: Could not import 'mask2d' from 'agent_mask2d'. Make sure the file exists.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # If verification_utils is missing, define a fallback simple checker
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if np.allclose(expected, actual):
                return True, "Arrays match"
            else:
                return False, f"Arrays mismatch. Expected shape {expected.shape}, Actual {actual.shape}"
        if expected == actual:
            return True, "Values match"
        return False, f"Expected {expected}, got {actual}"

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_mask2d.pkl']
    
    # Identify Data Files
    outer_data_path = None
    inner_data_path = None

    for p in data_paths:
        if "standard_data_mask2d.pkl" in p:
            outer_data_path = p
        elif "parent_function_mask2d" in p:
            inner_data_path = p

    if not outer_data_path:
        print("Error: Standard data file 'standard_data_mask2d.pkl' not found.")
        sys.exit(1)

    # 2. Load Outer Data (Arguments for mask2d)
    try:
        print(f"Loading outer data from {outer_data_path}...")
        outer_data = load_data(outer_data_path)
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output')
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Fix Random Seeds for Reproducibility
    # mask2d uses numpy.random, so we must try to align seeds if possible.
    # However, since the capture happened in the past, exact random reproduction 
    # might fail unless the capture environment had a fixed seed that we know.
    # The provided gen_data_code fixes seeds to 42. We do the same.
    np.random.seed(42)

    # 4. Execution Logic
    try:
        print("Executing mask2d with loaded arguments...")
        actual_result = mask2d(*outer_args, **outer_kwargs)
        
        # Scenario Check: Is the result a callable (factory pattern) or a direct value?
        # Based on the provided code for mask2d, it returns a numpy array 'mask', 
        # so it is NOT a factory pattern (Scenario A).
        
        # We handle Scenario B logic just in case the provided data implies it, 
        # but primarily we compare against outer_data['output'].

        if inner_data_path:
            # Factory Pattern (Scenario B) logic
            # If inner data exists, it implies mask2d returned a function that was then called.
            if not callable(actual_result):
                print(f"Error: Inner data exists ({inner_data_path}), expecting a callable from mask2d, but got {type(actual_result)}.")
                sys.exit(1)
            
            print(f"Loading inner data from {inner_data_path}...")
            inner_data = load_data(inner_data_path)
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_final_output = inner_data.get('output')

            print("Executing inner function...")
            final_result = actual_result(*inner_args, **inner_kwargs)
            
            check_target = expected_final_output
            actual_target = final_result
        else:
            # Simple Function (Scenario A) logic
            check_target = expected_outer_output
            actual_target = actual_result

    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    print("Verifying results...")
    
    # Note on Randomness: 
    # mask2d uses np.random.choice. Even with fixed seeds, differences in numpy versions 
    # or architecture can lead to different random draws.
    # If recursive_check fails strictly on values for random masks, we might need a looser check (e.g., shape/dtype).
    
    is_correct, msg = recursive_check(check_target, actual_target)

    if not is_correct:
        print(f"FAILURE: {msg}")
        # Secondary check for stochastic functions:
        # If the failure is value mismatch but shapes match, warn but maybe don't fail hard 
        # if we suspect randomness issues (unless strict reproducibility is guaranteed).
        # For this test, we assume strict reproducibility is expected via seed 42.
        print("Detailed comparison failed. This might be due to RNG differences if environments differ.")
        
        # Checking critical properties (shape and center mask)
        if isinstance(check_target, np.ndarray) and isinstance(actual_target, np.ndarray):
            if check_target.shape == actual_target.shape:
                print(f"Shape match confirmed: {check_target.shape}")
                # Check center region (deterministic part)
                # nx, ny are likely in outer_args
                try:
                    nx, ny = outer_args[0], outer_args[1]
                    center_r = outer_kwargs.get('center_r', 15)
                    # Simple heuristic check if center is 1s
                    cx, cy = int(nx/2), int(ny/2)
                    if actual_target[cx, cy] == 1.0:
                        print("Center pixel is correctly 1.0 (deterministic part likely correct).")
                    else:
                        print("Center pixel is NOT 1.0. Deterministic logic likely failed.")
                except:
                    pass
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()