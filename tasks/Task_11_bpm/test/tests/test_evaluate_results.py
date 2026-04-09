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
    Load pickled data file using dill.
    """
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"ERROR: Failed to load data from {file_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main test function for evaluate_results.
    """
    # Define data paths
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Categorize data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl")
        sys.exit(1)
    
    print(f"Found outer data file: {outer_path}")
    print(f"Found {len(inner_paths)} inner data file(s)")
    
    # Phase 1: Load outer data and execute the function
    print("\n" + "=" * 60)
    print("PHASE 1: Loading outer data and executing evaluate_results")
    print("=" * 60)
    
    try:
        outer_data = load_data(outer_path)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to extract outer data components")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("\nExecuting evaluate_results(*outer_args, **outer_kwargs)...")
        agent_result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify results
    print("\n" + "=" * 60)
    print("PHASE 2: Verification")
    print("=" * 60)
    
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Check if agent_result is callable
        if not callable(agent_result):
            print("WARNING: agent_result is not callable, treating as Scenario A")
            result = agent_result
            expected = outer_output
        else:
            print("agent_result is callable, proceeding with inner data execution")
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                
                try:
                    inner_data = load_data(inner_path)
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_output = inner_data.get('output', None)
                    
                    print(f"Inner args count: {len(inner_args)}")
                    print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to extract inner data components")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Execute the operator with inner data
                try:
                    print("\nExecuting agent_result(*inner_args, **inner_kwargs)...")
                    result = agent_result(*inner_args, **inner_kwargs)
                    expected = inner_output
                    print("Operator executed successfully.")
                    
                except Exception as e:
                    print(f"ERROR: Failed to execute agent_result operator")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify this inner result
                try:
                    passed, msg = recursive_check(expected, result)
                    
                    if not passed:
                        print(f"\nTEST FAILED for inner data: {inner_path}")
                        print(f"Failure message: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner data verification PASSED: {inner_path}")
                        
                except Exception as e:
                    print(f"ERROR: Verification failed with exception")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("\n" + "=" * 60)
            print("TEST PASSED")
            print("=" * 60)
            sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        result = agent_result
        expected = outer_output
    
    # Final verification for Scenario A (or Scenario B fallback)
    print("\nPerforming final verification...")
    
    try:
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print("\n" + "=" * 60)
            print("TEST FAILED")
            print("=" * 60)
            print(f"Failure message: {msg}")
            print(f"\nExpected type: {type(expected)}")
            print(f"Result type: {type(result)}")
            
            # Additional debug info for dictionaries
            if isinstance(expected, dict) and isinstance(result, dict):
                print(f"\nExpected keys: {list(expected.keys())}")
                print(f"Result keys: {list(result.keys())}")
                
                for key in expected.keys():
                    if key in result:
                        exp_val = expected[key]
                        res_val = result[key]
                        if exp_val != res_val:
                            print(f"\nMismatch at key '{key}':")
                            print(f"  Expected: {exp_val}")
                            print(f"  Got: {res_val}")
            
            sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print("TEST PASSED")
            print("=" * 60)
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR: Verification failed with exception")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()