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


def find_data_files(data_paths):
    """
    Categorize data files into outer (main function) and inner (closure/operator) data.
    """
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def load_data(file_path):
    """
    Load data from a pickle file using dill.
    """
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"ERROR: Failed to load data from {file_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        return None


def main():
    """
    Main test function for evaluate_results.
    """
    # Define data paths
    data_paths = ['/home/yjh/diff_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Categorize data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl")
        sys.exit(1)
    
    print(f"Found outer data file: {outer_path}")
    if inner_paths:
        print(f"Found inner data files: {inner_paths}")
    else:
        print("No inner data files found - using Scenario A (Simple Function)")
    
    # Phase 1: Load outer data and execute function
    print("\n=== Phase 1: Loading outer data and executing evaluate_results ===")
    
    outer_data = load_data(outer_path)
    if outer_data is None:
        print("ERROR: Failed to load outer data")
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    try:
        # Execute the target function
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Successfully executed evaluate_results")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify results
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Phase 2: Scenario B - Factory/Closure Pattern ===")
        
        # Check if result is callable (an operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, but inner data exists.")
            print("Proceeding with Scenario A verification instead.")
            # Fall back to Scenario A
            expected = outer_output
        else:
            agent_operator = result
            print("Result is callable - proceeding with inner data execution")
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                
                inner_data = load_data(inner_path)
                if inner_data is None:
                    print(f"ERROR: Failed to load inner data from {inner_path}")
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                    print("Successfully executed agent_operator with inner data")
                except Exception as e:
                    print(f"ERROR: Failed to execute agent_operator")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify results for this inner data
                print("\n=== Verification ===")
                try:
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Verification passed for {inner_path}")
                except Exception as e:
                    print(f"ERROR: Verification failed with exception")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("\nTEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple Function
        print("\n=== Phase 2: Scenario A - Simple Function ===")
        expected = outer_output
    
    # Final verification for Scenario A (or fallback)
    print("\n=== Verification ===")
    try:
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            print(f"Expected type: {type(expected)}")
            print(f"Result type: {type(result)}")
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                print(f"Expected value: {expected}")
                print(f"Result value: {result}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    except Exception as e:
        print(f"ERROR: Verification failed with exception")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()