import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent__deblur_core import _deblur_core
from verification_utils import recursive_check


def find_data_files():
    """Search for data files related to _deblur_core."""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    outer_path = None
    inner_paths = []
    
    # Search in current directory and common data subdirectories
    search_dirs = [data_dir, os.path.join(data_dir, 'data'), os.path.join(data_dir, 'test_data')]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for filename in os.listdir(search_dir):
            if filename.endswith('.pkl'):
                if 'standard_data__deblur_core.pkl' == filename:
                    outer_path = os.path.join(search_dir, filename)
                elif '_deblur_core' in filename and 'parent_function' in filename:
                    inner_paths.append(os.path.join(search_dir, filename))
                elif filename == 'standard_data__deblur_core.pkl':
                    outer_path = os.path.join(search_dir, filename)
    
    return outer_path, inner_paths


def test__deblur_core():
    """Test the _deblur_core function."""
    
    # Try to find data files
    outer_path, inner_paths = find_data_files()
    
    if outer_path and os.path.exists(outer_path):
        # Scenario: We have data files to test with
        print(f"Found outer data file: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"Error loading outer data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs from outer data
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        try:
            # Run the function
            result = _deblur_core(*outer_args, **outer_kwargs)
            print(f"Function executed successfully")
            print(f"Result type: {type(result)}")
            
            if callable(result) and inner_paths:
                # Scenario B: Factory pattern - result is an operator
                inner_path = inner_paths[0]
                print(f"Found inner data file: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output', None)
                
                # Execute the operator
                result = result(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully")
            
            # Compare results
            if expected_output is not None:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            else:
                print("No expected output to compare, but function executed successfully")
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"Error during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # No data files found - run a basic sanity test
        print("No data files found. Running basic sanity test...")
        
        try:
            # Create simple test data
            test_data = np.random.rand(64, 64).astype(np.float32)
            
            # Simple Gaussian kernel
            kernel_size = 5
            sigma = 1.0
            x = np.arange(kernel_size) - kernel_size // 2
            kernel_1d = np.exp(-x**2 / (2 * sigma**2))
            kernel = np.outer(kernel_1d, kernel_1d)
            kernel = kernel / kernel.sum()
            
            iteration = 5
            rule = 1  # Richardson-Lucy
            
            print(f"Test data shape: {test_data.shape}")
            print(f"Kernel shape: {kernel.shape}")
            print(f"Iterations: {iteration}")
            print(f"Rule: {rule}")
            
            # Run the function
            result = _deblur_core(test_data, kernel, iteration, rule)
            
            print(f"Result type: {type(result)}")
            print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            
            # Basic sanity checks
            assert result is not None, "Result should not be None"
            assert isinstance(result, np.ndarray), "Result should be numpy array"
            assert result.shape == test_data.shape, f"Result shape {result.shape} should match input shape {test_data.shape}"
            assert not np.isnan(result).any(), "Result should not contain NaN"
            assert not np.isinf(result).any(), "Result should not contain Inf"
            
            # Test with rule 2 (Landweber)
            result2 = _deblur_core(test_data, kernel, iteration, rule=2)
            assert result2 is not None, "Result for rule 2 should not be None"
            assert isinstance(result2, np.ndarray), "Result for rule 2 should be numpy array"
            
            print("Basic sanity test passed")
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"Basic sanity test failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    test__deblur_core()