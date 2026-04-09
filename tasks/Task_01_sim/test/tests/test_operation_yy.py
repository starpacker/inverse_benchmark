import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path to find the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_operation_yy import operation_yy
from verification_utils import recursive_check


def main():
    data_paths = []
    
    # Parse data paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if path and os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_operation_yy.pkl':
                outer_path = path
    
    # If no data files provided, we need to create a basic test
    if not data_paths or outer_path is None:
        print("No data files provided. Running basic functionality test...")
        
        try:
            # Test with a simple grid size
            test_gsize = (4, 4, 4)
            
            # Run the function
            result = operation_yy(test_gsize)
            
            # Basic validation - check that result is a numpy array with expected properties
            if result is None:
                print("FAILED: Function returned None")
                sys.exit(1)
            
            if not isinstance(result, np.ndarray):
                print(f"FAILED: Expected numpy array, got {type(result)}")
                sys.exit(1)
            
            # Check that the result has the expected shape (should match gsize)
            if result.shape != test_gsize:
                print(f"FAILED: Expected shape {test_gsize}, got {result.shape}")
                sys.exit(1)
            
            # Verify the result is complex (FFT output)
            if not np.iscomplexobj(result):
                print("WARNING: Expected complex output from FFT operation")
            
            # Additional sanity check - result should be non-negative real part 
            # since it's |FFT|^2 (conjugate multiplication)
            # Actually, yyfft * conj(yyfft) should give real positive values
            
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"FAILED: Exception during basic test: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario A/B: Load and test with provided data files
    try:
        # Phase 1: Load outer data and reconstruct operator/result
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Running operation_yy with args={outer_args}, kwargs={outer_kwargs}")
        agent_result = operation_yy(*outer_args, **outer_kwargs)
        
        # Phase 2: Check if this is a factory pattern (Scenario B) or simple function (Scenario A)
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            if not callable(agent_result):
                print(f"FAILED: Expected callable operator, got {type(agent_result)}")
                sys.exit(1)
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data['output']
                
                print(f"Executing operator with inner args={inner_args}, kwargs={inner_kwargs}")
                result = agent_result(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for {inner_path}")
        else:
            # Scenario A: Simple function
            expected = outer_data['output']
            result = agent_result
            
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"FAILED: Exception during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()