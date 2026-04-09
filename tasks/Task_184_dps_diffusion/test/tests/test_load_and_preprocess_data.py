import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    print(f"[Phase 1] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"  Args count: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the target function
    print("\n[Phase 2] Executing load_and_preprocess_data...")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\n[Scenario B] Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if callable(result):
            print("  Result is callable, proceeding with inner data execution...")
            
            for inner_path in inner_paths:
                print(f"\n  Loading inner data from: {inner_path}")
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"  ERROR: Failed to load inner data file: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                print(f"    Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"    Inner args count: {len(inner_args)}")
                print(f"    Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                try:
                    actual_result = result(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"  ERROR: Failed to execute operator with inner args: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify against inner expected output
                print("\n[Phase 3] Verifying inner execution results...")
                try:
                    passed, msg = recursive_check(inner_expected, actual_result)
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed: {msg}")
        else:
            # Result is not callable, treat as Scenario A with inner data for reference
            print("  Result is not callable, falling back to Scenario A verification")
            print("\n[Phase 3] Verifying results against outer expected output...")
            try:
                passed, msg = recursive_check(expected_output, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n[Scenario A] No inner data files found, verifying direct output...")
        print("\n[Phase 3] Verifying results against expected output...")
        
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()