import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_angular_spectrum_kernel import angular_spectrum_kernel

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for angular_spectrum_kernel."""
    
    # Data paths provided
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_angular_spectrum_kernel.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_angular_spectrum_kernel.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_angular_spectrum_kernel.pkl in data_paths")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing angular_spectrum_kernel with outer args/kwargs...")
        actual_result = angular_spectrum_kernel(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute angular_spectrum_kernel: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Check if the result is callable (an operator)
        if not callable(actual_result):
            print("WARNING: Result is not callable, treating as Scenario A instead")
            # Fall through to Scenario A logic
        else:
            # Process inner data files
            for inner_path in inner_paths:
                if not os.path.exists(inner_path):
                    print(f"ERROR: Inner data file does not exist: {inner_path}")
                    sys.exit(1)
                
                try:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                try:
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected_output = inner_data.get('output', None)
                except Exception as e:
                    print(f"ERROR: Failed to extract args/kwargs from inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                try:
                    print("Executing operator with inner args/kwargs...")
                    actual_result = actual_result(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR: Failed to execute operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify results
                try:
                    print("Verifying results...")
                    passed, msg = recursive_check(expected_output, actual_result)
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner test passed for: {inner_path}")
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
    
    # Scenario A: Simple Function (no inner paths or result not callable)
    print("Detected Scenario A: Simple Function")
    
    # The result from Phase 1 IS the result to compare
    result = actual_result
    expected = expected_output
    
    # Verify results
    try:
        print("Verifying results...")
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()