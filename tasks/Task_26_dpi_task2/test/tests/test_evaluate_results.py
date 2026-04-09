import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the necessary path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
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
    
    # Phase 1: Load outer data and execute the function
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer function name: {outer_data.get('func_name')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("\nExecuting evaluate_results with outer data...")
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (an operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, but inner paths exist. Treating as Scenario A.")
            expected = outer_output
        else:
            # Load inner data and execute the operator
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner function name: {inner_data.get('func_name')}")
            print(f"Number of inner args: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner data
            print("\nExecuting operator with inner data...")
            try:
                result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function execution")
        expected = outer_output
    
    # Phase 2: Verification
    print("\nVerifying results...")
    print(f"Expected type: {type(expected)}")
    print(f"Result type: {type(result)}")
    
    # Additional debug info for dictionaries
    if isinstance(expected, dict):
        print(f"Expected keys: {list(expected.keys())}")
    if isinstance(result, dict):
        print(f"Result keys: {list(result.keys())}")
    
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("\n" + "="*50)
        print("TEST PASSED")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("TEST FAILED")
        print("="*50)
        print(f"Failure message: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()