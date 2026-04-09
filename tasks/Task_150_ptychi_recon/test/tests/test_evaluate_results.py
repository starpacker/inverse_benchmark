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
    
    data_paths = ['/data/yjh/ptychi_recon_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Step 1: Categorize data files
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
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from {outer_path}")
        print(f"Outer data keys: {outer_data.keys()}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract outer args and kwargs
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute the function
    try:
        print("\n" + "="*60)
        print("Executing evaluate_results with outer data...")
        print("="*60)
        
        result = evaluate_results(*outer_args, **outer_kwargs)
        
        print(f"\nFunction executed successfully.")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine if this is a factory pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\nDetected factory/closure pattern (Scenario B)")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable from evaluate_results, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load and process inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"\nLoaded inner data from {inner_path}")
                print(f"Inner data keys: {inner_data.keys()}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner data
            try:
                print("\nExecuting agent_operator with inner data...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully.")
                print(f"Actual result type: {type(actual_result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"\nTEST FAILED for inner data {inner_path}")
                    print(f"Mismatch details: {msg}")
                    sys.exit(1)
                else:
                    print(f"\nVerification passed for inner data {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nDetected simple function pattern (Scenario A)")
        
        expected = outer_output
        actual_result = result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"\nTEST FAILED")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
            else:
                print(f"\nVerification passed")
        except Exception as e:
            print(f"ERROR: Failed during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*60)
    print("TEST PASSED")
    print("="*60)
    sys.exit(0)


if __name__ == "__main__":
    main()