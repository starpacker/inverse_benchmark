import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to the path to import the agent module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_back_diff import back_diff
from verification_utils import recursive_check


def find_data_files():
    """Find all relevant pickle files for testing."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    outer_path = None
    inner_paths = []
    
    # Search in current directory and common subdirectories
    search_dirs = [current_dir, os.path.join(current_dir, 'data'), os.path.join(current_dir, 'test_data')]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for filename in os.listdir(search_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(search_dir, filename)
                # Check for outer data file (exact match pattern)
                if filename == 'standard_data_back_diff.pkl':
                    outer_path = filepath
                # Check for inner data files (parent_function pattern)
                elif 'parent_function' in filename and 'back_diff' in filename:
                    inner_paths.append(filepath)
    
    return outer_path, inner_paths


def load_data(filepath):
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        data = dill.load(f)
    return data


def main():
    try:
        # Find data files
        outer_path, inner_paths = find_data_files()
        
        # If no data files found, create a simple test case
        if outer_path is None and not inner_paths:
            print("No data files found. Running basic functionality test...")
            
            # Create test data - 3D array
            test_data = np.random.rand(2, 3, 4).astype(np.float32)
            step = 0.1
            dim = 0
            
            try:
                result = back_diff(test_data, step, dim)
                print(f"Function executed successfully with shape: {result.shape}")
                print("TEST PASSED")
                sys.exit(0)
            except Exception as e:
                print(f"Function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        # Scenario A: Simple function test (only outer_path exists)
        if outer_path is not None and not inner_paths:
            print(f"Loading outer data from: {outer_path}")
            outer_data = load_data(outer_path)
            
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected = outer_data.get('output')
            
            print("Running back_diff function...")
            result = back_diff(*args, **kwargs)
            
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        
        # Scenario B: Factory/Closure pattern
        if outer_path is not None and inner_paths:
            print(f"Loading outer data from: {outer_path}")
            outer_data = load_data(outer_path)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print("Creating operator from outer data...")
            agent_operator = back_diff(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                # If result is not callable, compare directly
                expected = outer_data.get('output')
                result = agent_operator
                
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            
            # Test with inner data
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                inner_data = load_data(inner_path)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
        
        # If only inner paths exist (unlikely but handle it)
        if outer_path is None and inner_paths:
            print("ERROR: Found inner data files but no outer data file")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()