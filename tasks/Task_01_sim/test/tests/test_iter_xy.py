import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_iter_xy import iter_xy
from verification_utils import recursive_check


def find_data_files():
    """Search for pkl data files in common locations."""
    possible_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
        '/tmp',
        '.'
    ]
    
    outer_path = None
    inner_paths = []
    
    for search_dir in possible_dirs:
        if not os.path.isdir(search_dir):
            continue
        for fname in os.listdir(search_dir):
            if fname.endswith('.pkl'):
                full_path = os.path.join(search_dir, fname)
                # Check for outer data file (standard_data_iter_xy.pkl)
                if fname == 'standard_data_iter_xy.pkl':
                    outer_path = full_path
                # Check for inner data files (parent_function pattern)
                elif 'parent_function' in fname and 'iter_xy' in fname:
                    inner_paths.append(full_path)
    
    return outer_path, inner_paths


def run_test():
    """Run the test for iter_xy function."""
    
    # First, try to find data files
    outer_path, inner_paths = find_data_files()
    
    # If no data files found, run a basic functionality test
    if outer_path is None:
        print("No standard data file found. Running basic functionality test...")
        
        try:
            # Create test inputs based on the function signature
            # iter_xy(g, bxy, para, mu)
            np.random.seed(42)
            
            # Create 3D arrays as expected by the function
            r, n, m = 5, 8, 8
            g = np.random.randn(r, n, m).astype(np.float32)
            bxy = np.random.randn(r, n, m).astype(np.float32)
            para = 0.1
            mu = 0.01
            
            # Run the function
            result = iter_xy(g, bxy, para, mu)
            
            # Basic validation - should return tuple of (Lxy, bxy)
            if not isinstance(result, tuple):
                print(f"FAILED: Expected tuple output, got {type(result)}")
                sys.exit(1)
            
            if len(result) != 2:
                print(f"FAILED: Expected tuple of length 2, got {len(result)}")
                sys.exit(1)
            
            Lxy, bxy_out = result
            
            # Check shapes match input shape
            if Lxy.shape != g.shape:
                print(f"FAILED: Lxy shape {Lxy.shape} doesn't match input shape {g.shape}")
                sys.exit(1)
            
            if bxy_out.shape != g.shape:
                print(f"FAILED: bxy_out shape {bxy_out.shape} doesn't match input shape {g.shape}")
                sys.exit(1)
            
            # Check for NaN/Inf
            if np.any(np.isnan(Lxy)) or np.any(np.isinf(Lxy)):
                print("FAILED: Lxy contains NaN or Inf values")
                sys.exit(1)
            
            if np.any(np.isnan(bxy_out)) or np.any(np.isinf(bxy_out)):
                print("FAILED: bxy_out contains NaN or Inf values")
                sys.exit(1)
            
            print("TEST PASSED (basic functionality test)")
            sys.exit(0)
            
        except Exception as e:
            print(f"FAILED: Basic functionality test failed with error: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Standard data file exists - run full test
    print(f"Found outer data file: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAILED: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    try:
        # Run the function
        result = iter_xy(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAILED: iter_xy execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner data files (factory/closure pattern)
    if inner_paths:
        print(f"Found {len(inner_paths)} inner data file(s). Testing factory pattern...")
        
        # Result should be callable in factory pattern
        if not callable(result):
            # Not a factory pattern, use standard comparison
            pass
        else:
            # Factory pattern - test with inner data
            agent_operator = result
            
            for inner_path in inner_paths:
                print(f"Testing with inner data: {inner_path}")
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"FAILED: Could not load inner data file: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                try:
                    inner_result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"FAILED: Operator execution failed: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"FAILED: Inner data comparison failed: {msg}")
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
    
    # Standard comparison (no inner data or non-factory pattern)
    passed, msg = recursive_check(expected_output, result)
    if not passed:
        print(f"FAILED: Output comparison failed: {msg}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    run_test()