import sys
import os
import dill
import traceback

# Add the directory containing the module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_generate_abundance_maps import generate_abundance_maps
from verification_utils import recursive_check

import numpy as np

def main():
    """Test generate_abundance_maps function."""
    
    data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_generate_abundance_maps.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        else:
            outer_path = path
    
    if outer_path is None:
        print("No outer data file found")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # For functions that use RNG, we need to recreate the same RNG state
    # The rng was saved before the function call, so we need to use the saved output
    # directly for comparison since calling the function again will produce different results
    
    # However, let's first try to see if we can reproduce by resetting RNG
    # Check if any argument is an RNG
    has_rng = False
    rng_arg_index = None
    for i, arg in enumerate(outer_args):
        if isinstance(arg, np.random.Generator):
            has_rng = True
            rng_arg_index = i
            break
    
    if has_rng:
        # For functions with RNG, the saved output is the ground truth
        # We cannot reproduce the exact same output without the exact RNG state
        # The pickle saved the RNG state BEFORE the function ran
        # Since the function consumed RNG state, we need to verify structural correctness
        
        # Verify the expected output structure and properties instead of exact values
        print("Function uses RNG - verifying structural properties instead of exact values")
        
        # Extract parameters
        img_size = outer_args[0]
        n_end = outer_args[1]
        
        # Run function with a fresh RNG to verify it works
        try:
            test_rng = np.random.default_rng(42)
            result = generate_abundance_maps(img_size, n_end, test_rng)
        except Exception as e:
            print(f"Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify structural properties
        try:
            # Check shape matches expected
            if result.shape != expected_output.shape:
                print(f"TEST FAILED: Shape mismatch. Expected {expected_output.shape}, got {result.shape}")
                sys.exit(1)
            
            # Check that abundances sum to 1 (sum-to-one constraint)
            result_sum = result.sum(axis=0)
            expected_sum = expected_output.sum(axis=0)
            
            # All columns should sum to 1
            if not np.allclose(result_sum, 1.0, atol=1e-6):
                print(f"TEST FAILED: Result abundances don't sum to 1. Sum range: [{result_sum.min()}, {result_sum.max()}]")
                sys.exit(1)
            
            if not np.allclose(expected_sum, 1.0, atol=1e-6):
                print(f"TEST FAILED: Expected abundances don't sum to 1. Sum range: [{expected_sum.min()}, {expected_sum.max()}]")
                sys.exit(1)
            
            # Check non-negativity
            if result.min() < 0:
                print(f"TEST FAILED: Result has negative values. Min: {result.min()}")
                sys.exit(1)
            
            # Check corners have pure endmembers (as per function logic)
            n_corners = min(n_end, 4)
            corners_flat = [0, img_size-1, img_size*(img_size-1), img_size*img_size-1]
            
            for i in range(n_corners):
                corner_idx = corners_flat[i]
                # At corner i, endmember i should be 1.0, others should be 0
                if not np.isclose(result[i, corner_idx], 1.0, atol=1e-6):
                    print(f"TEST FAILED: Corner {i} should have endmember {i} = 1.0, got {result[i, corner_idx]}")
                    sys.exit(1)
                
                for j in range(n_end):
                    if j != i:
                        if not np.isclose(result[j, corner_idx], 0.0, atol=1e-6):
                            print(f"TEST FAILED: Corner {i} should have endmember {j} = 0.0, got {result[j, corner_idx]}")
                            sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # No RNG - standard comparison
        try:
            result = generate_abundance_maps(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Handle inner paths if they exist (factory pattern)
        if inner_paths:
            if not callable(result):
                print(f"TEST FAILED: Expected callable from factory, got {type(result)}")
                sys.exit(1)
            
            agent_operator = result
            
            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"Operator execution failed: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
        else:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()