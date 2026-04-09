import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def find_data_files(data_paths):
    """
    Analyze data paths to determine test scenario.
    Returns outer_path and inner_path (if exists).
    """
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def load_pickle_data(file_path):
    """
    Load data from a pickle file using dill.
    """
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"ERROR: Failed to load pickle file '{file_path}': {e}")
        traceback.print_exc()
        sys.exit(1)


def run_test():
    """
    Main test function implementing the test logic.
    """
    # Define data paths
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Step 1: Analyze data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl in data paths")
        sys.exit(1)
    
    print(f"Found outer data file: {outer_path}")
    if inner_paths:
        print(f"Found inner data files: {inner_paths}")
    else:
        print("No inner data files found - using Scenario A (Simple Function)")
    
    # Step 2: Load outer data
    print("\nLoading outer data...")
    outer_data = load_pickle_data(outer_path)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute the function
    try:
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("\nExecuting Scenario B: Factory/Closure Pattern")
            
            # Phase 1: Create the operator
            print("Phase 1: Creating operator...")
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            
            # Verify operator is callable
            if not callable(agent_operator):
                print(f"ERROR: forward_operator did not return a callable. Got: {type(agent_operator)}")
                sys.exit(1)
            
            print(f"Operator created successfully: {type(agent_operator)}")
            
            # Phase 2: Execute with inner data
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                inner_data = load_pickle_data(inner_path)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {inner_path}")
            
            print("\nTEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple Function
            print("\nExecuting Scenario A: Simple Function")
            
            # Execute the function directly
            print("Executing forward_operator...")
            result = forward_operator(*outer_args, **outer_kwargs)
            
            expected = outer_output
            
            # Verify results
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("\nTEST PASSED")
                sys.exit(0)
    
    except Exception as e:
        print(f"ERROR: Exception during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_test()