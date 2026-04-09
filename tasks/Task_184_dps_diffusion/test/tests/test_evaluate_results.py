import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add path for imports
sys.path.insert(0, '/data/yjh/dps_diffusion_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Step 1: Analyze data paths to determine test strategy
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
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute function with outer data
    try:
        print("\nExecuting evaluate_results with outer data...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function returned result of type: {type(result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine if this is a factory pattern or simple function
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("\nDetected factory/closure pattern - loading inner data...")
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            try:
                print("\nExecuting agent_operator with inner data...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Agent operator returned result of type: {type(actual_result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"\nTEST FAILED for inner path {inner_path}")
                    print(f"Mismatch details: {msg}")
                    sys.exit(1)
                else:
                    print(f"\nVerification passed for inner path: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("\nDetected simple function pattern - comparing direct output...")
        
        expected = outer_output
        actual_result = result
        
        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"\nTEST FAILED")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
            else:
                print("\nTEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()