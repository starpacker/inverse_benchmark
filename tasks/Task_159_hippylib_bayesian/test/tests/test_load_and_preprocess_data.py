import sys
import os
import dill
import numpy as np
import traceback

# Add the necessary paths
sys.path.insert(0, '/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/hippylib_bayesian_sandbox_sandbox')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Import scipy.sparse for handling sparse matrices
from scipy import sparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two scipy sparse matrices."""
    if type(expected) != type(actual):
        return False, f"Type mismatch: {type(expected)} vs {type(actual)}"
    
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
    
    # Convert to dense for comparison or compare as arrays
    expected_coo = expected.tocoo()
    actual_coo = actual.tocoo()
    
    # Check if they have the same number of non-zeros (approximately)
    if abs(expected_coo.nnz - actual_coo.nnz) > 0:
        # They might still be equal if some zeros are explicit
        pass
    
    # Convert to dense and compare
    expected_dense = expected.toarray()
    actual_dense = actual.toarray()
    
    if not np.allclose(expected_dense, actual_dense, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(expected_dense - actual_dense))
        return False, f"Sparse matrix values differ, max diff: {max_diff}"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """Custom recursive check that handles scipy sparse matrices."""
    
    # Handle scipy sparse matrices
    if sparse.issparse(expected) or sparse.issparse(actual):
        if not sparse.issparse(expected) or not sparse.issparse(actual):
            return False, f"At {path}: One is sparse, one is not"
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"At {path}: Shape mismatch {expected.shape} vs {actual.shape}"
        if expected.dtype != actual.dtype:
            # Try to compare anyway if numeric
            pass
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"At {path}: Array values differ, max diff: {max_diff}"
        return True, "Arrays match"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            missing = set(expected.keys()) - set(actual.keys())
            extra = set(actual.keys()) - set(expected.keys())
            return False, f"At {path}: Key mismatch. Missing: {missing}, Extra: {extra}"
        
        for key in expected.keys():
            passed, msg = custom_recursive_check(
                expected[key], actual[key], rtol, atol, path=f"{path}['{key}']"
            )
            if not passed:
                return False, msg
        return True, "Dictionaries match"
    
    # Handle lists
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False, f"At {path}: List length mismatch {len(expected)} vs {len(actual)}"
        
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Lists match"
    
    # Handle tuples
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False, f"At {path}: Tuple length mismatch {len(expected)} vs {len(actual)}"
        
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}({i})")
            if not passed:
                return False, msg
        return True, "Tuples match"
    
    # Handle numeric scalars
    if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
        if not np.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            return False, f"At {path}: Scalar mismatch {expected} vs {actual}"
        return True, "Scalars match"
    
    # Handle strings and other types
    if type(expected) != type(actual):
        return False, f"At {path}: Type mismatch {type(expected)} vs {type(actual)}"
    
    # Default comparison
    try:
        if expected != actual:
            return False, f"At {path}: Value mismatch {expected} vs {actual}"
    except ValueError:
        # This can happen with arrays that can't be compared directly
        return False, f"At {path}: Cannot compare values of type {type(expected)}"
    
    return True, "Match"


def main():
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args: {len(outer_args)} positional arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Successfully executed load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario and get expected output
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        # Load inner data
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Successfully loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # The result from Phase 1 should be callable
        if not callable(result):
            print("ERROR: Expected callable operator from Phase 1")
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        
        try:
            result = result(*inner_args, **inner_kwargs)
            print("Successfully executed inner operator")
        except Exception as e:
            print(f"ERROR: Failed to execute inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data['output']
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        expected = outer_data['output']
    
    # Verification using custom recursive check that handles sparse matrices
    try:
        passed, msg = custom_recursive_check(expected, result)
        
        if not passed:
            print(f"ERROR: Verification failed: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()