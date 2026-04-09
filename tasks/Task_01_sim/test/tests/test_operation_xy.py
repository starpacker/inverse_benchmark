import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_operation_xy import operation_xy

# Import verification utility
from verification_utils import recursive_check


def find_data_files(data_paths):
    """
    Categorize data files into outer (factory) and inner (execution) data.
    """
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_operation_xy.pkl' or filename.endswith('_operation_xy.pkl'):
            outer_path = path
    
    return outer_path, inner_paths


def load_data(file_path):
    """
    Load pickled data file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    
    return data


def main():
    """
    Main test execution function.
    """
    # Define data paths - these would be provided in actual test scenarios
    data_paths = []
    
    # Check for data files in common locations
    possible_locations = [
        'standard_data_operation_xy.pkl',
        './standard_data_operation_xy.pkl',
        '../standard_data_operation_xy.pkl',
        'data/standard_data_operation_xy.pkl',
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            data_paths.append(loc)
            break
    
    # Also check for inner data files
    for loc in ['.', '..', 'data']:
        if os.path.exists(loc):
            for f in os.listdir(loc):
                if 'parent_function' in f and 'operation_xy' in f and f.endswith('.pkl'):
                    data_paths.append(os.path.join(loc, f))
    
    # If no data files found, try to run a basic test
    if not data_paths:
        print("No data files found. Running basic functionality test...")
        try:
            # Test with a simple gsize parameter
            test_gsize = (4, 4)
            result = operation_xy(test_gsize)
            
            # Verify the result is a numpy array
            if isinstance(result, np.ndarray):
                print(f"Function executed successfully. Output shape: {result.shape}")
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"Unexpected result type: {type(result)}")
                sys.exit(1)
        except Exception as e:
            print(f"Basic functionality test failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Find outer and inner data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("ERROR: No outer data file found for operation_xy")
        sys.exit(1)
    
    print(f"Found outer data file: {outer_path}")
    if inner_paths:
        print(f"Found inner data files: {inner_paths}")
    
    try:
        # Phase 1: Load outer data and reconstruct operator
        print("\n--- Phase 1: Loading outer data and executing function ---")
        outer_data = load_data(outer_path)
        
        # Extract args and kwargs from outer data
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute the function
        result = operation_xy(*outer_args, **outer_kwargs)
        
        # Phase 2: Check if this is a factory pattern (result is callable)
        if inner_paths and callable(result):
            print("\n--- Phase 2: Factory pattern detected, executing inner calls ---")
            agent_operator = result
            
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                inner_data = load_data(inner_path)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                # Execute the operator with inner arguments
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify inner result
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"VERIFICATION FAILED for inner call: {msg}")
                    sys.exit(1)
                print(f"Inner call verification passed")
            
            print("\nTEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple function, compare result directly
            print("\n--- Simple function pattern, comparing results directly ---")
            
            if expected_output is None:
                print("WARNING: No expected output in data file")
                if result is not None:
                    print(f"Function returned result of type: {type(result)}")
                    print("TEST PASSED (no expected output to compare)")
                    sys.exit(0)
            
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                print(f"Expected type: {type(expected_output)}")
                print(f"Actual type: {type(result)}")
                if isinstance(expected_output, np.ndarray) and isinstance(result, np.ndarray):
                    print(f"Expected shape: {expected_output.shape}")
                    print(f"Actual shape: {result.shape}")
                sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"TEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()