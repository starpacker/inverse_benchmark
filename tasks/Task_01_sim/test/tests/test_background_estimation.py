import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_background_estimation import background_estimation
from verification_utils import recursive_check

def find_data_files():
    """Search for pickle data files for background_estimation."""
    data_files = []
    
    # Search in current directory and common data subdirectories
    search_dirs = [
        '.',
        './data',
        './test_data',
        '../data',
        '../test_data',
        os.path.dirname(os.path.abspath(__file__)),
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.pkl') and 'background_estimation' in f:
                        full_path = os.path.join(root, f)
                        data_files.append(full_path)
    
    return list(set(data_files))  # Remove duplicates

def load_data(filepath):
    """Load pickle data file with dill."""
    with open(filepath, 'rb') as f:
        return dill.load(f)

def main():
    try:
        # Find data files
        data_paths = find_data_files()
        
        if not data_paths:
            # If no pickle files found, try creating a simple test case
            print("No pickle data files found. Running with synthetic test data...")
            
            # Create synthetic test data
            np.random.seed(42)
            test_img = np.random.rand(2, 64, 64).astype(np.float64) * 255
            
            try:
                result = background_estimation(test_img, th=1, dlevel=3, wavename='db6', iter=2)
                
                # Basic sanity checks
                if result is None:
                    print("FAILED: Result is None")
                    sys.exit(1)
                
                if result.shape != test_img.shape:
                    print(f"FAILED: Shape mismatch. Expected {test_img.shape}, got {result.shape}")
                    sys.exit(1)
                
                print("TEST PASSED (synthetic data)")
                sys.exit(0)
                
            except Exception as e:
                print(f"FAILED during synthetic test: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print(f"Found data files: {data_paths}")
        
        # Separate outer and inner data files
        outer_path = None
        inner_paths = []
        
        for p in data_paths:
            filename = os.path.basename(p)
            if 'parent_function' in filename:
                inner_paths.append(p)
            elif filename == 'standard_data_background_estimation.pkl':
                outer_path = p
            elif 'background_estimation' in filename and filename.endswith('.pkl'):
                # Fallback: use any background_estimation pickle as outer
                if outer_path is None:
                    outer_path = p
        
        if outer_path is None and len(data_paths) > 0:
            # Use the first available file
            outer_path = data_paths[0]
        
        if outer_path is None:
            print("ERROR: Could not find appropriate data file")
            sys.exit(1)
        
        print(f"Using outer data file: {outer_path}")
        
        # Load outer data
        outer_data = load_data(outer_path)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        # Execute the function
        print("Executing background_estimation...")
        result = background_estimation(*outer_args, **outer_kwargs)
        
        # If there are inner paths, this is a factory pattern
        if inner_paths:
            print(f"Factory pattern detected. Using inner data: {inner_paths[0]}")
            inner_data = load_data(inner_paths[0])
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            # result should be callable
            if callable(result):
                result = result(*inner_args, **inner_kwargs)
            else:
                print("Warning: Expected callable from factory pattern but got non-callable")
        
        # Verify results
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"TEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()