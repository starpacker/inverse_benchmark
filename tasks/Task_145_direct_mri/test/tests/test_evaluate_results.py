import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/direct_mri_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Classify paths into outer (main function) and inner (nested/closure calls)
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Validate we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl in data_paths")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute the main function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_expected = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the main function
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Successfully executed evaluate_results")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A (simple) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        # The result should be callable (an operator/closure)
        if not callable(result):
            print(f"WARNING: Expected callable from evaluate_results, got {type(result)}")
            # Fall back to Scenario A comparison
            print("Falling back to Scenario A comparison")
            expected = outer_expected
        else:
            print(f"Got callable operator: {result}")
            
            # Load inner data and execute the operator
            inner_path = inner_paths[0]  # Use the first inner path
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            try:
                result = result(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        expected = outer_expected
    
    # Phase 2: Verification
    print("\n" + "=" * 50)
    print("VERIFICATION PHASE")
    print("=" * 50)
    
    print(f"Result type: {type(result)}")
    print(f"Expected type: {type(expected)}")
    
    # Use recursive_check for comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: recursive_check raised an exception")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("\n" + "=" * 50)
        print("TEST PASSED")
        print("=" * 50)
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("TEST FAILED")
        print("=" * 50)
        print(f"Mismatch details: {msg}")
        
        # Additional debug info
        if isinstance(result, dict) and isinstance(expected, dict):
            print("\nExpected keys:", list(expected.keys()))
            print("Result keys:", list(result.keys()))
            for key in expected:
                if key in result:
                    if expected[key] != result[key]:
                        print(f"  Key '{key}': expected={expected[key]}, got={result[key]}")
                else:
                    print(f"  Key '{key}' missing in result")
        
        sys.exit(1)

if __name__ == "__main__":
    main()