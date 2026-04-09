import sys
import os
import dill
import torch
import numpy as np
import traceback
import glob

# Import the target function
from agent_operation_yz import operation_yz
from verification_utils import recursive_check


def find_data_files():
    """Search for data files in common locations."""
    search_paths = [
        ".",
        "./data",
        "../data",
        "/home/yjh/sim_sandbox",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    
    outer_path = None
    inner_paths = []
    
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue
            
        # Search for outer data file
        pattern_outer = os.path.join(base_path, "**", "*standard_data_operation_yz.pkl")
        for f in glob.glob(pattern_outer, recursive=True):
            if "parent_function" not in f:
                outer_path = f
                break
        
        # Search for inner data files
        pattern_inner = os.path.join(base_path, "**", "*parent_function*operation_yz*.pkl")
        inner_paths.extend(glob.glob(pattern_inner, recursive=True))
        
        if outer_path:
            break
    
    return outer_path, inner_paths


def test_with_data_files(outer_path, inner_paths):
    """Test using loaded data files."""
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Phase 1: Create the operator/result
    agent_result = operation_yz(*outer_args, **outer_kwargs)
    
    if inner_paths:
        # Scenario B: Factory pattern
        print(f"\nScenario B: Factory pattern detected")
        print(f"Agent result type: {type(agent_result)}")
        
        if not callable(agent_result):
            print("ERROR: Expected callable from operation_yz but got non-callable")
            sys.exit(1)
        
        for inner_path in inner_paths:
            print(f"\nTesting with inner data: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data['output']
            
            result = agent_result(*inner_args, **inner_kwargs)
            
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
            print(f"Inner test passed!")
    else:
        # Scenario A: Simple function
        print(f"\nScenario A: Simple function")
        expected = outer_data['output']
        result = agent_result
        
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
    
    return True


def test_without_data_files():
    """Test the function with generated inputs when no data files exist."""
    print("No data files found. Running tests with generated inputs...")
    
    # Test case 1: Small grid size
    gsize = (4, 4, 4)
    print(f"\nTest 1: gsize = {gsize}")
    
    try:
        result = operation_yz(gsize)
        print(f"Result type: {type(result)}")
        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        
        # Verify the result manually
        delta_yz = np.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
        yzfft = np.fft.fftn(delta_yz, gsize) * np.conj(np.fft.fftn(delta_yz, gsize))
        expected = yzfft
        
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        print("Test 1 passed!")
        
    except Exception as e:
        print(f"Test 1 failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test case 2: Different grid size
    gsize2 = (8, 8, 8)
    print(f"\nTest 2: gsize = {gsize2}")
    
    try:
        result2 = operation_yz(gsize2)
        print(f"Result type: {type(result2)}")
        print(f"Result shape: {result2.shape}")
        
        # Verify
        delta_yz = np.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
        yzfft2 = np.fft.fftn(delta_yz, gsize2) * np.conj(np.fft.fftn(delta_yz, gsize2))
        expected2 = yzfft2
        
        passed, msg = recursive_check(expected2, result2)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        print("Test 2 passed!")
        
    except Exception as e:
        print(f"Test 2 failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test case 3: Non-cubic grid
    gsize3 = (4, 6, 8)
    print(f"\nTest 3: gsize = {gsize3}")
    
    try:
        result3 = operation_yz(gsize3)
        print(f"Result shape: {result3.shape}")
        
        delta_yz = np.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
        yzfft3 = np.fft.fftn(delta_yz, gsize3) * np.conj(np.fft.fftn(delta_yz, gsize3))
        expected3 = yzfft3
        
        passed, msg = recursive_check(expected3, result3)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        print("Test 3 passed!")
        
    except Exception as e:
        print(f"Test 3 failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    return True


def main():
    print("=" * 60)
    print("Testing operation_yz")
    print("=" * 60)
    
    # Try to find data files
    outer_path, inner_paths = find_data_files()
    
    print(f"Outer data file: {outer_path}")
    print(f"Inner data files: {inner_paths}")
    
    try:
        if outer_path and os.path.exists(outer_path):
            # Test with data files
            test_with_data_files(outer_path, inner_paths)
        else:
            # Test with generated inputs
            test_without_data_files()
        
        print("\n" + "=" * 60)
        print("TEST PASSED")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        print(f"\nTEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()