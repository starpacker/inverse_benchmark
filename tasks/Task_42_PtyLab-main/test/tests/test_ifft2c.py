import sys
import os
import dill
import numpy as np
import scipy.fft
import traceback

# Soft import for torch to handle environments where it's missing
try:
    import torch
except ImportError:
    torch = None

# Add the directory containing the target function to path
sys.path.append('/data/yjh/PtyLab-main_sandbox/run_code')

# Import target function
try:
    from agent_ifft2c import ifft2c
except ImportError:
    # Fallback definition if file import fails, ensuring self-contained test capability
    def ifft2c(x):
        """Centered 2D Inverse FFT."""
        return scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.ifftshift(x)))

# Import verification utils
from verification_utils import recursive_check

def run_test():
    data_paths = ['/data/yjh/PtyLab-main_sandbox/run_code/std_data/standard_data_ifft2c.pkl']
    
    # 1. Identify Data Files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if 'standard_data_ifft2c.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_ifft2c' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_ifft2c.pkl not found in paths.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    print(f"Loaded outer data from {outer_path}")
    
    # 3. Execute Target Function
    # Logic: ifft2c is a function that returns a result. It is not a factory returning a closure.
    # Therefore, we treat this as Scenario A (Simple Function).
    # If inner_path existed, it would imply a factory pattern, but looking at the function def, it's standard.
    
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        # Execute
        actual_result = ifft2c(*outer_args, **outer_kwargs)
        
        # Check if the result is a callable (Scenario B detection just in case)
        if callable(actual_result) and not isinstance(actual_result, (np.ndarray, list, tuple)):
             if inner_path:
                 print("Detected Factory Pattern. Loading inner data...")
                 with open(inner_path, 'rb') as f:
                     inner_data = dill.load(f)
                 inner_args = inner_data.get('args', [])
                 inner_kwargs = inner_data.get('kwargs', {})
                 expected_output = inner_data.get('output') # Update expected to inner output
                 
                 # Execute Inner
                 actual_result = actual_result(*inner_args, **inner_kwargs)
             else:
                 # It returned a callable but we have no inner data to test it with.
                 # We must compare the callable object itself (unlikely to succeed) or fail.
                 print("Warning: Function returned a callable but no inner data found to execute it.")

    except Exception as e:
        traceback.print_exc()
        print(f"Execution failed: {e}")
        sys.exit(1)

    # 4. Verify Results
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        print(f"Verification process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()