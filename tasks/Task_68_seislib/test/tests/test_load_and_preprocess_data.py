import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to the path
sys.path.insert(0, '/data/yjh/seislib_sandbox_sandbox')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Import scipy.sparse for handling sparse matrices
from scipy.sparse import issparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices."""
    if not issparse(expected) or not issparse(actual):
        return False, "One of the matrices is not sparse"
    
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
    
    # Convert to same format for comparison
    expected_csr = expected.tocsr()
    actual_csr = actual.tocsr()
    
    # Compare data, indices, and indptr
    if not np.allclose(expected_csr.data, actual_csr.data, rtol=rtol, atol=atol):
        return False, "Sparse matrix data values differ"
    
    if not np.array_equal(expected_csr.indices, actual_csr.indices):
        return False, "Sparse matrix indices differ"
    
    if not np.array_equal(expected_csr.indptr, actual_csr.indptr):
        return False, "Sparse matrix indptr differ"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """Custom recursive check that handles sparse matrices."""
    # Handle sparse matrices
    if issparse(expected) and issparse(actual):
        return compare_sparse_matrices(expected, actual, rtol, atol)
    elif issparse(expected) or issparse(actual):
        return False, f"Type mismatch at {path}: one is sparse, other is not"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Dict keys differ at {path}: {set(expected.keys())} vs {set(actual.keys())}"
        for k in expected.keys():
            passed, msg = custom_recursive_check(expected[k], actual[k], rtol, atol, path=f"{path}['{k}']")
            if not passed:
                return False, msg
        return True, "Match"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: {expected.shape} vs {actual.shape}"
        if expected.dtype != actual.dtype:
            # Allow dtype differences if values match
            pass
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            return False, f"Array values differ at {path}"
        return True, "Match"
    
    # Handle lists
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False, f"List length mismatch at {path}: {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Match"
    
    # Handle tuples
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False, f"Tuple length mismatch at {path}: {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Match"
    
    # Handle scalars
    if isinstance(expected, (int, float, np.integer, np.floating)) and isinstance(actual, (int, float, np.integer, np.floating)):
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"Scalar mismatch at {path}: {expected} vs {actual}"
        return True, "Match"
    
    # Handle strings and other types
    if type(expected) != type(actual):
        return False, f"Type mismatch at {path}: {type(expected)} vs {type(actual)}"
    
    if expected != actual:
        return False, f"Value mismatch at {path}: {expected} vs {actual}"
    
    return True, "Match"


def main():
    data_paths = ['/data/yjh/seislib_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Execute the function
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and we have inner paths)
    if inner_paths and callable(result):
        print("Detected factory/closure pattern")
        # Load inner data and execute
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            result = result(*inner_args, **inner_kwargs)
            print("Successfully executed inner function")
        except Exception as e:
            print(f"ERROR: Failed to execute inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Detected simple function pattern")
    
    # Verify results using custom recursive check that handles sparse matrices
    try:
        passed, msg = custom_recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()