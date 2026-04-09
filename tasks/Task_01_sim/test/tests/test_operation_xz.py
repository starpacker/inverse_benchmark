import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_operation_xz import operation_xz
from verification_utils import recursive_check


def find_data_files(data_dir='data'):
    """Find all relevant .pkl files in the data directory."""
    pkl_files = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.pkl') and 'operation_xz' in f:
                pkl_files.append(os.path.join(data_dir, f))
    return pkl_files


def load_data(file_path):
    """Load data from a pickle file using dill."""
    with open(file_path, 'rb') as f:
        return dill.load(f)


def run_test_with_data(data_paths):
    """Run tests using provided data files."""
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif 'operation_xz' in basename:
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    print(f"Loading outer data from: {outer_path}")
    outer_data = load_data(outer_path)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Running operation_xz with args: {outer_args}, kwargs: {outer_kwargs}")
    
    try:
        result = operation_xz(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern
    if inner_paths and callable(result):
        # Scenario B: Factory pattern
        print("Detected factory/closure pattern")
        agent_operator = result
        
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            inner_data = load_data(inner_path)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED for {inner_path}: {msg}")
                sys.exit(1)
            print(f"Inner test passed for {inner_path}")
    else:
        # Scenario A: Simple function
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


def run_test_without_data():
    """Run tests without pre-recorded data by testing function behavior directly."""
    print("No data files found. Running direct function validation...")
    
    try:
        # Test with a simple grid size
        gsize = (4, 4, 4)
        result = operation_xz(gsize)
        
        # Verify the result has expected properties
        # The function should return an FFT result based on delta_xz
        delta_xz = np.array([[[1, -1]], [[-1, 1]]], dtype='float32')
        expected = np.fft.fftn(delta_xz, gsize) * np.conj(np.fft.fftn(delta_xz, gsize))
        
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        # Additional test with different grid size
        gsize2 = (8, 8, 8)
        result2 = operation_xz(gsize2)
        expected2 = np.fft.fftn(delta_xz, gsize2) * np.conj(np.fft.fftn(delta_xz, gsize2))
        
        passed2, msg2 = recursive_check(expected2, result2)
        if not passed2:
            print(f"TEST FAILED for gsize {gsize2}: {msg2}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Test execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main test entry point."""
    # Try to find data files
    data_paths = find_data_files('data')
    
    # Also check current directory
    if not data_paths:
        data_paths = find_data_files('.')
    
    if data_paths:
        print(f"Found {len(data_paths)} data file(s)")
        run_test_with_data(data_paths)
    else:
        run_test_without_data()


if __name__ == '__main__':
    main()