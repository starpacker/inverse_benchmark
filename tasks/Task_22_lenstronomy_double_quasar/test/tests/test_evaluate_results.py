import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lenstronomy_double_quasar_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and run the function
    try:
        print("\n--- Phase 1: Loading outer data ---")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\n--- Phase 1: Executing evaluate_results ---")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n--- Scenario B: Factory/Closure Pattern ---")
        
        # Check if result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"\n--- Loading inner data: {inner_path} ---")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator
                print("\n--- Executing operator with inner data ---")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("\n--- Comparing results ---")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed to process inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("\n--- Scenario A: Simple Function ---")
        
        expected = outer_output
        
        # Compare results
        print("\n--- Comparing results ---")
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()