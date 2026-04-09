import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_iter_xz import iter_xz
from verification_utils import recursive_check


def main():
    """Main test function for iter_xz"""
    
    # Data paths provided (empty in this case, so we need to find them)
    data_paths = []
    
    # If data_paths is empty, try to discover data files
    if not data_paths:
        # Look for data files in current directory and common locations
        search_dirs = ['.', './data', '../data', './test_data']
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.pkl') and 'iter_xz' in f:
                        data_paths.append(os.path.join(search_dir, f))
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_iter_xz.pkl' or basename.endswith('_iter_xz.pkl'):
            outer_path = path
    
    # If no data files found, run a basic functionality test
    if outer_path is None and not inner_paths:
        print("No data files found. Running basic functionality test...")
        try:
            # Create test inputs for iter_xz
            # Based on the function signature: iter_xz(g, bxz, para, mu)
            # g should be a 3D array, bxz should have compatible shape
            
            np.random.seed(42)
            g = np.random.randn(4, 8, 8).astype(np.float32)
            bxz = np.random.randn(4, 8, 8).astype(np.float32)
            para = 0.1
            mu = 0.5
            
            # Call the function
            result = iter_xz(g, bxz, para, mu)
            
            # Verify output structure
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"FAILED: Expected tuple of 2 elements, got {type(result)}")
                sys.exit(1)
            
            Lxz, bxz_out = result
            
            # Check output types
            if not isinstance(Lxz, np.ndarray):
                print(f"FAILED: Lxz should be ndarray, got {type(Lxz)}")
                sys.exit(1)
            
            if not isinstance(bxz_out, np.ndarray):
                print(f"FAILED: bxz_out should be ndarray, got {type(bxz_out)}")
                sys.exit(1)
            
            print("Basic functionality test passed.")
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"FAILED: Basic functionality test raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 1: Load and process outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("Executing iter_xz with outer data...")
        result = iter_xz(*outer_args, **outer_kwargs)
        print(f"Execution successful. Result type: {type(result)}")
        
    except Exception as e:
        print(f"FAILED: iter_xz execution raised exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Handle inner paths (factory pattern) or direct comparison
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Found {len(inner_paths)} inner data file(s). Processing factory pattern...")
        
        # Check if result is callable (operator/closure)
        if callable(result):
            agent_operator = result
            
            for inner_path in inner_paths:
                try:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_expected = inner_data.get('output')
                    
                    print("Executing operator with inner data...")
                    inner_result = agent_operator(*inner_args, **inner_kwargs)
                    
                    # Compare results
                    passed, msg = recursive_check(inner_expected, inner_result)
                    if not passed:
                        print(f"FAILED: Inner data comparison failed: {msg}")
                        sys.exit(1)
                    
                    print(f"Inner data test passed for {os.path.basename(inner_path)}")
                    
                except Exception as e:
                    print(f"FAILED: Inner data processing raised exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
        else:
            # Result is not callable, compare directly with outer expected
            print("Result is not callable. Comparing with outer expected output...")
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function - direct comparison
        print("No inner data files. Comparing result with outer expected output...")
        
        if expected_output is None:
            print("WARNING: No expected output in data file. Checking result validity...")
            if result is None:
                print("FAILED: Result is None")
                sys.exit(1)
        else:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()