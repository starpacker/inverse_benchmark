import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_operation_zz import operation_zz
from verification_utils import recursive_check


def main():
    """Main test function for operation_zz."""
    
    # Data paths provided (empty in this case)
    data_paths = []
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            filename = os.path.basename(path)
            if 'parent_function' in filename:
                inner_paths.append(path)
            elif filename == 'standard_data_operation_zz.pkl':
                outer_path = path
    
    # If no data paths provided, look for standard locations
    if outer_path is None:
        standard_outer = 'standard_data_operation_zz.pkl'
        if os.path.exists(standard_outer):
            outer_path = standard_outer
    
    # Also check for inner paths with pattern matching
    if not inner_paths:
        import glob
        inner_patterns = glob.glob('standard_data_parent_function_operation_zz_*.pkl')
        inner_paths = [p for p in inner_patterns if os.path.exists(p)]
    
    # Scenario: No data files exist - create a simple test case
    if outer_path is None or not os.path.exists(outer_path):
        print("No data files found. Running direct function test...")
        try:
            # Test with a simple grid size
            test_gsize = (8, 8, 8)
            result = operation_zz(test_gsize)
            
            # Verify the result is a numpy array with correct shape
            if not isinstance(result, np.ndarray):
                print(f"FAILED: Expected numpy array, got {type(result)}")
                sys.exit(1)
            
            if result.shape != test_gsize:
                print(f"FAILED: Expected shape {test_gsize}, got {result.shape}")
                sys.exit(1)
            
            # Verify result is complex (FFT output)
            if not np.iscomplexobj(result):
                print(f"FAILED: Expected complex array from FFT")
                sys.exit(1)
            
            print("Direct function test passed.")
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"FAILED: Exception during direct test: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
    except Exception as e:
        print(f"FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 1: Execute the function with outer args
    try:
        print("Phase 1: Executing operation_zz with outer arguments...")
        agent_result = operation_zz(*outer_args, **outer_kwargs)
        print(f"Phase 1 completed. Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"FAILED: Exception during Phase 1 execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if we have inner data (Scenario B) or not (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Scenario B: Found {len(inner_paths)} inner data file(s)")
        
        # Verify the result is callable for closure pattern
        if not callable(agent_result):
            print(f"FAILED: Expected callable from operation_zz, got {type(agent_result)}")
            sys.exit(1)
        
        # Test with each inner data file
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print("Phase 2: Executing agent_result with inner arguments...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"FAILED for {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASSED for {inner_path}")
                    
            except Exception as e:
                print(f"FAILED: Exception during inner data processing ({inner_path}): {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("Scenario A: Simple function test (no inner data)")
        
        result = agent_result
        expected = outer_output
        
        if expected is None:
            print("WARNING: No expected output in data file. Verifying basic properties...")
            # Basic verification for FFT result
            if isinstance(result, np.ndarray):
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"FAILED: Expected numpy array, got {type(result)}")
                sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"FAILED: Exception during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()