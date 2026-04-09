import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check


def main():
    # Data paths provided (empty list means we need to find them)
    data_paths = []
    
    # If data_paths is empty, search for pkl files in current directory and common locations
    if not data_paths:
        search_dirs = ['.', './data', '../data', './test_data', '../test_data']
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.pkl') and 'load_and_preprocess_data' in f:
                        data_paths.append(os.path.join(search_dir, f))
    
    # Identify outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
        elif 'load_and_preprocess_data' in filename and 'parent_function' not in filename:
            # Could be the outer path with different naming
            if outer_path is None:
                outer_path = path
    
    # Check if we have any data files to work with
    if outer_path is None and not inner_paths:
        print("No data files found for testing load_and_preprocess_data")
        print(f"Searched paths: {data_paths}")
        # If no test data, we'll create a minimal test
        print("Running minimal functional test without serialized data...")
        
        try:
            # Create a temporary test image file
            import tempfile
            from skimage import io
            
            # Create a simple test image
            test_img = np.random.rand(64, 64).astype('float32') * 255
            
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name
                io.imsave(tmp_path, test_img.astype('uint8'))
            
            try:
                # Test the function
                result, scaler = load_and_preprocess_data(tmp_path, up_sample=0)
                
                # Basic sanity checks
                assert isinstance(result, np.ndarray), "Result should be a numpy array"
                assert isinstance(scaler, (int, float, np.number)), "Scaler should be a number"
                assert result.shape == test_img.shape, "Output shape should match input shape"
                assert result.max() <= 1.0, "Output should be normalized"
                
                print("Minimal functional test PASSED")
                sys.exit(0)
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"Minimal functional test FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    try:
        # Phase 1: Load outer data and reconstruct operator/result
        if outer_path and os.path.exists(outer_path):
            print(f"Loading outer data from: {outer_path}")
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            outer_output = outer_data.get('output', None)
            
            print(f"Outer args: {len(outer_args)} positional arguments")
            print(f"Outer kwargs: {list(outer_kwargs.keys())}")
            
            # Execute the function with outer args
            try:
                agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
                print("Function executed successfully")
            except Exception as e:
                print(f"Error executing load_and_preprocess_data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Phase 2: Check if this is a factory pattern (inner paths exist)
            if inner_paths:
                # Scenario B: Factory/Closure Pattern
                print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
                
                # Check if result is callable (operator)
                if callable(agent_result):
                    agent_operator = agent_result
                    
                    for inner_path in inner_paths:
                        print(f"Loading inner data from: {inner_path}")
                        with open(inner_path, 'rb') as f:
                            inner_data = dill.load(f)
                        
                        inner_args = inner_data.get('args', ())
                        inner_kwargs = inner_data.get('kwargs', {})
                        expected = inner_data.get('output', None)
                        
                        # Execute the operator with inner args
                        try:
                            actual_result = agent_operator(*inner_args, **inner_kwargs)
                        except Exception as e:
                            print(f"Error executing operator: {e}")
                            traceback.print_exc()
                            sys.exit(1)
                        
                        # Compare results
                        passed, msg = recursive_check(expected, actual_result)
                        if not passed:
                            print(f"TEST FAILED for inner data {inner_path}")
                            print(f"Mismatch details: {msg}")
                            sys.exit(1)
                        else:
                            print(f"Inner test passed for {inner_path}")
                    
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print("Result is not callable but inner paths exist - treating as Scenario A")
            
            # Scenario A: Simple function (or non-callable result with inner paths)
            print("Running Scenario A: Simple function comparison")
            expected = outer_output
            result = agent_result
            
            passed, msg = recursive_check(expected, result)
            if not passed:
                print("TEST FAILED")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        
        else:
            # No outer path but maybe only inner paths exist
            if inner_paths:
                print("Only inner paths found, attempting direct execution...")
                for inner_path in inner_paths:
                    print(f"Loading data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        data = dill.load(f)
                    
                    args = data.get('args', ())
                    kwargs = data.get('kwargs', {})
                    expected = data.get('output', None)
                    
                    try:
                        result = load_and_preprocess_data(*args, **kwargs)
                    except Exception as e:
                        print(f"Error executing function: {e}")
                        traceback.print_exc()
                        sys.exit(1)
                    
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"TEST FAILED for {inner_path}")
                        print(f"Mismatch details: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Test passed for {inner_path}")
                
                print("TEST PASSED")
                sys.exit(0)
            else:
                print("No valid data files found")
                sys.exit(1)
                
    except Exception as e:
        print(f"Unexpected error during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()