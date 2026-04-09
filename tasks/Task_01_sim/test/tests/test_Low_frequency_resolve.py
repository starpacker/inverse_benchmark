import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_Low_frequency_resolve import Low_frequency_resolve
from verification_utils import recursive_check


def main():
    # Data paths provided
    data_paths = []
    
    # If no data paths provided, we need to handle this case
    if not data_paths:
        print("No data paths provided. Looking for standard data files in current directory.")
        # Try to find data files in common locations
        possible_paths = [
            'standard_data_Low_frequency_resolve.pkl',
            './standard_data_Low_frequency_resolve.pkl',
            '../standard_data_Low_frequency_resolve.pkl',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                data_paths.append(p)
                break
        
        # Also look for inner data files
        for root, dirs, files in os.walk('.'):
            for f in files:
                if 'parent_function_Low_frequency_resolve' in f and f.endswith('.pkl'):
                    data_paths.append(os.path.join(root, f))
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_Low_frequency_resolve.pkl' or basename.endswith('_Low_frequency_resolve.pkl'):
            outer_path = path
    
    # If still no outer path, create a test case
    if outer_path is None:
        print("No standard data file found. Creating synthetic test case.")
        try:
            # Create synthetic test data based on the function signature
            # Low_frequency_resolve(coeffs, dlevel)
            # coeffs[0] = cAn (low frequency approximation)
            # coeffs[i] for i > 0 = (cH, cV, cD) tuples (detail coefficients)
            
            dlevel = 2
            cAn = np.random.rand(4, 4)
            coeffs = [cAn]
            
            for i in range(1, dlevel + 1):
                cH = np.random.rand(4, 4)
                cV = np.random.rand(4, 4)
                cD = np.random.rand(4, 4)
                coeffs.append((cH, cV, cD))
            
            # Run the function
            result = Low_frequency_resolve(coeffs, dlevel)
            
            # Verify the result structure
            # Expected: vec[0] = cAn, vec[1:] = tuples of zeros with same shape
            assert len(result) == dlevel + 1, f"Expected {dlevel + 1} elements, got {len(result)}"
            assert np.array_equal(result[0], cAn), "First element should be cAn"
            
            for i in range(1, dlevel + 1):
                t = result[i]
                assert isinstance(t, tuple) and len(t) == 3, f"Element {i} should be a 3-tuple"
                for j, arr in enumerate(t):
                    assert arr.shape == coeffs[i][0].shape, f"Shape mismatch at level {i}, component {j}"
                    assert np.allclose(arr, 0), f"Expected zeros at level {i}, component {j}"
            
            print("Synthetic test case PASSED")
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"Synthetic test failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args: {len(outer_args)} arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing Low_frequency_resolve...")
        agent_result = Low_frequency_resolve(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"Failed to execute Low_frequency_resolve: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Found {len(inner_paths)} inner data file(s). Testing factory pattern.")
        
        # Verify the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from factory, got {type(agent_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                print(f"Executing operator with inner args...")
                actual_result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(inner_expected, actual_result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for {inner_path}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {inner_path}")
                    
            except Exception as e:
                print(f"Failed to process inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("No inner data files. Testing as simple function.")
        
        if expected_output is None:
            print("WARNING: No expected output in data file. Performing basic validation.")
            # Basic validation based on function logic
            if agent_result is not None:
                print("Function returned a result (not None)")
                print("TEST PASSED")
                sys.exit(0)
            else:
                print("ERROR: Function returned None")
                sys.exit(1)
        
        # Compare results
        passed, msg = recursive_check(expected_output, agent_result)
        
        if not passed:
            print("VERIFICATION FAILED")
            print(f"Message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()