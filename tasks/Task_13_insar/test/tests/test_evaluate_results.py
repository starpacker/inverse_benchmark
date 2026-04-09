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
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (contains 'parent_function')
        if 'parent_function' in basename:
            inner_paths.append(path)
        # Check if this is the outer data file (exact pattern match)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Validate we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    
    try:
        # Phase 1: Load outer data and execute function
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Execute the target function
        print("\n=== Executing evaluate_results ===")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle based on scenario
    if is_factory_pattern:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Verify the result is callable (an operator)
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        print("Agent operator is callable")
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\n--- Processing inner data: {inner_path} ---")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Execute the operator with inner data
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            print("\n=== Verifying results ===")
            passed, msg = recursive_check(expected, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"Inner data verification passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("\n=== Scenario A: Simple Function ===")
        
        expected = outer_output
        actual_result = result
        
        print(f"Expected type: {type(expected)}")
        print(f"Actual result type: {type(actual_result)}")
        
        # Verify results
        print("\n=== Verifying results ===")
        passed, msg = recursive_check(expected, actual_result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print(f"Verification message: {msg}")
        print("\nTEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()