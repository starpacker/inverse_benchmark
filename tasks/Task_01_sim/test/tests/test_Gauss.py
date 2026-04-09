import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_Gauss import Gauss
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = []
    
    # If no data paths provided, we need to search for them
    if not data_paths:
        # Search in current directory and common data directories
        search_dirs = ['.', './data', '../data', './test_data', '../test_data']
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.pkl') and 'Gauss' in f:
                        data_paths.append(os.path.join(search_dir, f))
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_Gauss.pkl' or filename.endswith('_Gauss.pkl'):
            outer_path = path
    
    # If no data files found, create a simple test case
    if outer_path is None and not inner_paths:
        print("No data files found. Running basic functionality test...")
        try:
            # Test with a simple sigma value
            test_sigma = 2.0
            result = Gauss(test_sigma)
            
            # Basic validation - should return a numpy array that is a valid PSF
            if not isinstance(result, np.ndarray):
                print(f"FAILED: Expected numpy array, got {type(result)}")
                sys.exit(1)
            
            # PSF should sum to approximately 1
            if not np.isclose(result.sum(), 1.0, rtol=1e-5):
                print(f"FAILED: PSF should sum to 1, got {result.sum()}")
                sys.exit(1)
            
            # Test with 2D sigma
            test_sigma_2d = [2.0, 3.0]
            result_2d = Gauss(test_sigma_2d)
            
            if not isinstance(result_2d, np.ndarray):
                print(f"FAILED: Expected numpy array for 2D, got {type(result_2d)}")
                sys.exit(1)
            
            if not np.isclose(result_2d.sum(), 1.0, rtol=1e-5):
                print(f"FAILED: 2D PSF should sum to 1, got {result_2d.sum()}")
                sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"FAILED: Basic test raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    try:
        # Phase 1: Load outer data and reconstruct operator/result
        if outer_path is not None:
            print(f"Loading outer data from: {outer_path}")
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Executing Gauss with args: {outer_args}, kwargs: {outer_kwargs}")
            agent_result = Gauss(*outer_args, **outer_kwargs)
            
            # Check if this is Scenario B (factory pattern) with inner data
            if inner_paths:
                # Scenario B: Factory/Closure Pattern
                print("Detected factory/closure pattern with inner data files")
                
                # Verify agent_result is callable
                if not callable(agent_result):
                    print(f"FAILED: Expected callable operator, got {type(agent_result)}")
                    sys.exit(1)
                
                # Process each inner data file
                for inner_path in inner_paths:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output')
                    
                    print(f"Executing operator with inner args: {inner_args}, kwargs: {inner_kwargs}")
                    result = agent_result(*inner_args, **inner_kwargs)
                    
                    # Compare results
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"FAILED: {msg}")
                        sys.exit(1)
                    print(f"Inner test passed for {inner_path}")
            else:
                # Scenario A: Simple function call
                print("Detected simple function pattern")
                expected = outer_data.get('output')
                result = agent_result
                
                # Compare results
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAILED: {msg}")
                    sys.exit(1)
        
        elif inner_paths:
            # Only inner paths exist - unusual but handle it
            print("Only inner data files found, processing directly...")
            for inner_path in inner_paths:
                print(f"Loading data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Executing Gauss with args: {inner_args}, kwargs: {inner_kwargs}")
                result = Gauss(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAILED: {msg}")
                    sys.exit(1)
                print(f"Test passed for {inner_path}")
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"FAILED: Exception during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()