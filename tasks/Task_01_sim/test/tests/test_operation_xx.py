import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_operation_xx import operation_xx
from verification_utils import recursive_check

def find_data_file(filename):
    """Search for the data file in common locations."""
    search_paths = [
        filename,
        os.path.join('data', filename),
        os.path.join('test_data', filename),
        os.path.join('..', filename),
        os.path.join('..', 'data', filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), 'data', filename),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    # Also search recursively in current directory
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    
    return None

def load_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return dill.load(f)

def main():
    print("=" * 60)
    print("Testing operation_xx")
    print("=" * 60)
    
    # Find the standard data file
    outer_filename = 'standard_data_operation_xx.pkl'
    outer_path = find_data_file(outer_filename)
    
    if outer_path is None:
        print(f"ERROR: Could not find {outer_filename}")
        print("Searching in current directory and common subdirectories")
        
        # List all .pkl files for debugging
        print("\nAvailable .pkl files:")
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.pkl'):
                    print(f"  {os.path.join(root, f)}")
        sys.exit(1)
    
    print(f"Found data file: {outer_path}")
    
    # Look for inner data files (factory/closure pattern)
    inner_path = None
    inner_pattern = 'standard_data_parent_function_operation_xx'
    for root, dirs, files in os.walk('.'):
        for f in files:
            if inner_pattern in f and f.endswith('.pkl'):
                inner_path = os.path.join(root, f)
                break
        if inner_path:
            break
    
    try:
        # Phase 1: Load outer data and execute function
        print("\nPhase 1: Loading outer data and executing operation_xx")
        outer_data = load_data(outer_path)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"  Args: {outer_args}")
        print(f"  Kwargs: {outer_kwargs}")
        
        # Execute the function
        result = operation_xx(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine expected output and compare
        if inner_path is not None:
            # Scenario B: Factory/Closure Pattern
            print(f"\nPhase 2: Found inner data file: {inner_path}")
            print("  Using Factory/Closure pattern")
            
            inner_data = load_data(inner_path)
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            
            # The result from phase 1 should be callable
            if callable(result):
                print("  Executing the returned operator with inner args")
                result = result(*inner_args, **inner_kwargs)
            
            expected = inner_data.get('output')
        else:
            # Scenario A: Simple function
            print("\nPhase 2: Using simple function pattern")
            expected = outer_data.get('output')
        
        # Phase 3: Verification
        print("\nPhase 3: Verifying results")
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\n" + "=" * 60)
            print("TEST PASSED")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("TEST FAILED")
            print("=" * 60)
            print(f"Mismatch details: {msg}")
            print(f"\nExpected type: {type(expected)}")
            print(f"Result type: {type(result)}")
            if isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
                print(f"Expected shape: {expected.shape}")
                print(f"Result shape: {result.shape}")
                print(f"Expected dtype: {expected.dtype}")
                print(f"Result dtype: {result.dtype}")
            sys.exit(1)
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED - Exception occurred")
        print("=" * 60)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()