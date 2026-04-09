import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/stride_sandbox_sandbox')
sys.path.insert(0, '/data/yjh/stride_sandbox_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Import scipy sparse for comparison
from scipy.sparse import issparse, csr_matrix


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices."""
    if not issparse(expected) or not issparse(actual):
        return False, "One or both matrices are not sparse"
    
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    
    # Convert to same format for comparison
    exp_csr = expected.tocsr()
    act_csr = actual.tocsr()
    
    # Compare data arrays
    if not np.allclose(exp_csr.data, act_csr.data, rtol=rtol, atol=atol):
        return False, "Sparse matrix data values differ"
    
    # Compare indices
    if not np.array_equal(exp_csr.indices, act_csr.indices):
        return False, "Sparse matrix indices differ"
    
    # Compare indptr
    if not np.array_equal(exp_csr.indptr, act_csr.indptr):
        return False, "Sparse matrix indptr differ"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """Custom recursive check that handles sparse matrices properly."""
    
    # Handle sparse matrices
    if issparse(expected) or issparse(actual):
        if not (issparse(expected) and issparse(actual)):
            return False, f"{path}: One is sparse, the other is not"
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"{path}: Shape mismatch {expected.shape} vs {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            return False, f"{path}: Array values differ"
        return True, "Match"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"{path}: Dict keys differ. Expected {set(expected.keys())}, got {set(actual.keys())}"
        for k in expected.keys():
            passed, msg = custom_recursive_check(expected[k], actual[k], rtol, atol, path=f"{path}['{k}']")
            if not passed:
                return False, msg
        return True, "Match"
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"{path}: Length mismatch {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Match"
    
    # Handle scalars and other types
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(actual, (int, float, np.integer, np.floating)):
            if np.isnan(expected) and np.isnan(actual):
                return True, "Match"
            if not np.isclose(expected, actual, rtol=rtol, atol=atol):
                return False, f"{path}: Scalar mismatch {expected} vs {actual}"
            return True, "Match"
    
    # Direct comparison for other types
    try:
        if expected == actual:
            return True, "Match"
        else:
            return False, f"{path}: Value mismatch {expected} vs {actual}"
    except ValueError:
        # This can happen with arrays in == comparison
        return False, f"{path}: Cannot compare values directly"


def main():
    # Data paths provided
    data_paths = ['/data/yjh/stride_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args: {len(outer_args)} positional arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern or simple function
    if inner_paths:
        # Factory/Closure pattern - need to execute the returned operator
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        if not callable(result):
            print(f"ERROR: Expected callable from load_and_preprocess_data, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
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
            expected_output = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = custom_recursive_check(expected_output, result)
                if not passed:
                    print(f"ERROR: Verification failed: {msg}")
                    sys.exit(1)
                print("Verification passed for inner data")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Simple function pattern
        print("Detected simple function pattern (no inner data files)")
        
        # Verify result using custom check that handles sparse matrices
        try:
            passed, msg = custom_recursive_check(expected_output, result)
            if not passed:
                print(f"ERROR: Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()