def rliter(yk, data, otf):
    rliter_val = np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))
    return rliter_val


import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_rliter import rliter
from verification_utils import recursive_check


def generate_test_data():
    """Generate test data for rliter function when no data files are available."""
    np.random.seed(42)
    
    # Create small test arrays for the RL iteration
    # yk: current estimate
    # data: observed data
    # otf: optical transfer function
    shape = (8, 8, 8)
    
    # Generate positive real data (typical for deconvolution)
    yk = np.abs(np.random.randn(*shape)) + 0.1  # Current estimate (positive)
    data = np.abs(np.random.randn(*shape)) + 0.1  # Observed data (positive)
    
    # OTF is typically complex (Fourier transform of PSF)
    otf = np.fft.fftn(np.abs(np.random.randn(*shape)) + 0.1)
    
    # Compute expected output
    expected_output = np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))
    
    return {
        'args': (yk, data, otf),
        'kwargs': {},
        'output': expected_output
    }


def run_test():
    """Main test function."""
    
    # Since data_paths is empty, generate our own test data
    data_paths = []
    
    # Check if any data files exist
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if path.endswith('standard_data_rliter.pkl') and 'parent_function' not in path:
            outer_path = path
        elif 'parent_function' in path and 'rliter' in path:
            inner_path = path
    
    if outer_path and os.path.exists(outer_path):
        # Scenario: Load from file
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        expected = outer_data.get('output')
        
        # Execute function
        result = rliter(*args, **kwargs)
        
        if inner_path and os.path.exists(inner_path):
            # This would be for closure/factory pattern - not applicable here
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            result = result(*inner_args, **inner_kwargs)
    else:
        # No data files available - generate test data
        print("No data files found. Generating test data...")
        test_data = generate_test_data()
        
        args = test_data['args']
        kwargs = test_data['kwargs']
        expected = test_data['output']
        
        print(f"Running rliter with generated data...")
        print(f"  yk shape: {args[0].shape}, dtype: {args[0].dtype}")
        print(f"  data shape: {args[1].shape}, dtype: {args[1].dtype}")
        print(f"  otf shape: {args[2].shape}, dtype: {args[2].dtype}")
        
        # Execute function
        result = rliter(*args, **kwargs)
        
        print(f"  result shape: {result.shape}, dtype: {result.dtype}")
    
    # Verify results
    print("Verifying results...")
    passed, msg = recursive_check(expected, result)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"TEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)