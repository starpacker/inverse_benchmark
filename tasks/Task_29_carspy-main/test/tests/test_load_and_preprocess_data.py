import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Fix seeds globally to maximize reproducibility where possible
np.random.seed(42)
torch.manual_seed(42)

def main():
    """
    Unit test for load_and_preprocess_data.
    
    Strategy:
    1. Identify data files (Scenario A or B).
    2. Execute the function with inputs loaded from pickle.
    3. Verify outputs. Since the function contains random noise generation,
       we must handle stochastic mismatch gracefully.
    """
    data_paths = ['/data/yjh/carspy-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Separate Outer (Agent Construction) and Inner (Closure Execution) data
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        elif 'load_and_preprocess_data.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_load_and_preprocess_data.pkl not found in paths.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        print(f"Loading primary data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load pickle file {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. Execute Function
    try:
        print("Executing load_and_preprocess_data with loaded arguments...")
        
        # Check if noise is involved to anticipate stochastic behavior
        # Signature: (raw_signal, nu_axis, noise_level=0.0)
        noise_level = 0.0
        if len(outer_args) >= 3:
            noise_level = outer_args[2]
        elif 'noise_level' in outer_kwargs:
            noise_level = outer_kwargs['noise_level']
            
        print(f"  Detected noise_level: {noise_level}")

        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Scenario Handling (A: Direct Result vs B: Factory)
    
    # Check if result is a closure/factory (Callable but not the final data)
    # The reference code suggests it returns (signal, axis), not a callable.
    # However, we support the structure if it turned out to be a factory.
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, list, tuple)):
        # Scenario B: Factory Pattern
        if not inner_paths:
            print("Function returned a callable (Scenario B), but no inner data files found to test it.")
            sys.exit(1)
            
        print("Function returned an operator. Testing inner execution paths...")
        operator = actual_result
        
        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                i_expected = inner_data.get('output')
                
                i_result = operator(*i_args, **i_kwargs)
                
                # Check results
                passed, msg = recursive_check(i_expected, i_result)
                if not passed:
                    print(f"Inner test failed for {os.path.basename(ip)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"Inner execution failed for {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)
                
        print("All inner paths passed.")
        sys.exit(0)

    else:
        # Scenario A: Simple Function (Direct Result)
        print("Verifying results (Direct Output)...")
        
        # We need custom verification because of random noise
        # Standard recursive_check is strict.
        
        # 1. Check Structure (tuple of 2 items)
        if not isinstance(actual_result, (tuple, list)) or len(actual_result) != 2:
            print(f"FAILED: Expected tuple of length 2, got {type(actual_result)}")
            sys.exit(1)

        actual_signal, actual_axis = actual_result
        expected_signal, expected_axis = expected_output

        # 2. Strict check on deterministic axis (nu_axis)
        passed_axis, msg_axis = recursive_check(expected_axis, actual_axis)
        if not passed_axis:
            print(f"TEST FAILED: Axis mismatch (deterministic output). {msg_axis}")
            sys.exit(1)

        # 3. Relaxed check on signal
        if noise_level > 0:
            print("  Stochastic noise detected. Performing statistical sanity check instead of strict equality.")
            
            # Ensure shapes match
            if actual_signal.shape != expected_signal.shape:
                 print(f"TEST FAILED: Shape mismatch. Expected {expected_signal.shape}, got {actual_signal.shape}")
                 sys.exit(1)

            # Check that signals are in the same ballpark (statistical properties)
            # Since noise is random, exact values won't match, but mean/std and range should be similar
            diff = np.abs(actual_signal - expected_signal)
            mean_diff = np.mean(diff)
            
            # Heuristic: If normalized, values are 0-1. 
            # With noise_level e.g., 0.1, differences can be large per pixel, but average difference shouldn't be massive.
            # Using a generous tolerance for random noise variance.
            stochastic_tol = 0.5 
            
            if mean_diff > stochastic_tol:
                print(f"TEST FAILED: Mean difference {mean_diff} exceeds stochastic tolerance {stochastic_tol}.")
                sys.exit(1)
            else:
                print(f"  Stochastic check passed: Mean difference {mean_diff:.4f} is within tolerance.")

        else:
            # If no noise, should be deterministic or close to it
            passed_sig, msg_sig = recursive_check(expected_signal, actual_signal, tol=1e-5)
            if not passed_sig:
                print(f"TEST FAILED: Signal mismatch (deterministic case). {msg_sig}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()