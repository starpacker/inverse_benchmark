import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check


def find_data_files(data_paths):
    """
    Categorize data files into outer (main function) and inner (closure/operator) paths.
    """
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def load_data_file(file_path):
    """
    Load a pickle file using dill.
    """
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"FAILED: Error loading data file '{file_path}': {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    # Define data paths
    data_paths = ['/home/yjh/fpm_inr_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Step 1: Categorize data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("FAILED: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"Found outer data file: {outer_path}")
    if inner_paths:
        print(f"Found inner data files: {inner_paths}")
    else:
        print("No inner data files found - using Scenario A (Simple Function)")
    
    # Step 2: Load outer data
    print("\nPhase 1: Loading outer data and executing function...")
    outer_data = load_data_file(outer_path)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Step 3: Execute the target function
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"FAILED: Error executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("\nPhase 2: Executing operator with inner data (Scenario B)...")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAILED: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            inner_data = load_data_file(inner_path)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: Error executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            passed, msg = recursive_check(expected_output, actual_result)
            
            if not passed:
                print(f"FAILED: Verification failed for {inner_path}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"Verification passed for {inner_path}")
    else:
        # Scenario A: Simple Function
        print("\nPhase 2: Verifying direct function output (Scenario A)...")
        
        result = agent_result
        expected = outer_output
        
        if expected is None:
            print("WARNING: No expected output found in outer data. Skipping comparison.")
            print("TEST PASSED (no expected output to compare)")
            sys.exit(0)
        
        # Compare results
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"FAILED: Verification failed")
            print(f"Message: {msg}")
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()