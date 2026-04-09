import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/reconformer_mri_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
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
    
    # Phase 1: Load outer data and run function
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
    
    print(f"Outer data loaded successfully.")
    print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"  Number of args: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("\nExecuting evaluate_results with outer args/kwargs...")
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Function executed successfully.")
    
    # Check if this is a factory pattern (result is callable and we have inner paths)
    if len(inner_paths) > 0 and callable(result) and not isinstance(result, type):
        # Scenario B: Factory/Closure Pattern
        print("\nDetected factory/closure pattern. Processing inner data...")
        
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_output = inner_data.get('output')
            
            print(f"Inner data loaded successfully.")
            print(f"  Function name: {inner_data.get('func_name', 'unknown')}")
            print(f"  Number of args: {len(inner_args)}")
            print(f"  Kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            print("\nExecuting operator with inner args/kwargs...")
            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("\nComparing results...")
            try:
                passed, msg = recursive_check(inner_output, actual_result)
            except Exception as e:
                print(f"ERROR: Failed during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - compare result directly with expected output
        print("\nSimple function pattern detected. Comparing output directly...")
        
        expected = outer_output
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)

if __name__ == "__main__":
    main()