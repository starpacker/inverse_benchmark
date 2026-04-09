import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_compute_psf_fft import compute_psf_fft
from verification_utils import recursive_check


def main():
    """Main test function for compute_psf_fft."""
    
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_compute_psf_fft.pkl']
    
    # Step 1: Categorize data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_psf_fft.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_psf_fft.pkl)")
        sys.exit(1)
    
    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract outer args and kwargs
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute compute_psf_fft
    try:
        # The function takes a 'setup' dictionary as input
        result = compute_psf_fft(*outer_args, **outer_kwargs)
        print(f"Successfully executed compute_psf_fft")
        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_psf_fft: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Check if this is a factory pattern (inner paths exist)
    if inner_paths:
        print(f"\nFactory pattern detected. Found {len(inner_paths)} inner data file(s).")
        
        # Verify the result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"\nLoaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"VERIFICATION FAILED for {inner_path}:")
                    print(msg)
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner data: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Simple function pattern - compare result directly with expected output
        print("\nSimple function pattern detected.")
        
        if expected_output is None:
            print("WARNING: No expected output found in data file")
            # If no expected output, we just verify the function ran without error
            print("TEST PASSED (execution only, no output comparison)")
            sys.exit(0)
        
        # Verify result
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print("VERIFICATION FAILED:")
                print(msg)
                sys.exit(1)
            else:
                print("Verification passed for function output")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()