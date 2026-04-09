import sys
import os
import traceback

import dill
import torch
import numpy as np

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Successfully loaded outer data from {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_expected = outer_data.get('output', None)
    
    print(f"[INFO] Outer args count: {len(outer_args)}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("[INFO] Successfully executed evaluate_results")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Detected Scenario B (Factory/Closure pattern)")
        
        # Check if result is callable
        if not callable(result):
            print(f"[INFO] Result is not callable, treating as Scenario A instead")
            # Fall back to Scenario A
            expected = outer_expected
        else:
            print("[INFO] Result is callable, proceeding with inner data execution")
            
            # Load inner data and execute
            inner_path = inner_paths[0]  # Use first inner path
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Successfully loaded inner data from {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"[INFO] Inner args count: {len(inner_args)}")
            print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the callable result with inner data
            try:
                result = result(*inner_args, **inner_kwargs)
                print("[INFO] Successfully executed inner function")
            except Exception as e:
                print(f"ERROR: Failed to execute inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[INFO] Detected Scenario A (Simple function)")
        expected = outer_expected
    
    # Comparison phase
    print("[INFO] Comparing results...")
    
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()