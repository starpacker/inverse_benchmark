import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def find_data_files(data_paths):
    """
    Categorize data files into outer (factory) and inner (execution) paths.
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


def load_data(file_path):
    """
    Load data from a pickle file using dill.
    """
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"FAILED: Error loading data from {file_path}: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    # Define data paths
    data_paths = ['/home/yjh/lensless_admm_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Find and categorize data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("FAILED: No outer data file found (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    print(f"Loading outer data from: {outer_path}")
    outer_data = load_data(outer_path)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    try:
        # Execute forward_operator with the outer data
        print("Executing forward_operator...")
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAILED: Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be a callable operator
        print(f"Detected factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
        if not callable(result):
            print(f"FAILED: Expected callable operator from forward_operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            inner_data = load_data(inner_path)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                print("Executing the created operator with inner data...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: Error executing operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            print("Verifying inner result...")
            passed, msg = recursive_check(inner_expected, inner_result)
            
            if not passed:
                print(f"FAILED: Inner result verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed for: {os.path.basename(inner_path)}")
    
    else:
        # Scenario A: Simple function - compare result directly
        print("Detected simple function pattern (Scenario A)")
        
        # Verify the result against expected output
        print("Verifying result...")
        passed, msg = recursive_check(expected_output, result)
        
        if not passed:
            print(f"FAILED: Result verification failed")
            print(f"Message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()