import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_build_diffusion_matrix import build_diffusion_matrix
from verification_utils import recursive_check

# Import scipy.sparse for matrix comparison
from scipy import sparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Compare two scipy sparse matrices.
    Returns (passed, message).
    """
    if not sparse.issparse(expected) or not sparse.issparse(actual):
        return False, f"Expected both to be sparse matrices, got {type(expected)} and {type(actual)}"
    
    # Convert both to CSR format for consistent comparison
    expected_csr = expected.tocsr()
    actual_csr = actual.tocsr()
    
    # Check shapes
    if expected_csr.shape != actual_csr.shape:
        return False, f"Shape mismatch: expected {expected_csr.shape}, got {actual_csr.shape}"
    
    # Check number of non-zeros
    if expected_csr.nnz != actual_csr.nnz:
        return False, f"Number of non-zeros mismatch: expected {expected_csr.nnz}, got {actual_csr.nnz}"
    
    # Compare the data, indices, and indptr arrays
    if not np.allclose(expected_csr.data, actual_csr.data, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(expected_csr.data - actual_csr.data))
        return False, f"Data values mismatch, max difference: {max_diff}"
    
    if not np.array_equal(expected_csr.indices, actual_csr.indices):
        return False, "Column indices mismatch"
    
    if not np.array_equal(expected_csr.indptr, actual_csr.indptr):
        return False, "Index pointer mismatch"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Custom recursive check that handles sparse matrices.
    """
    # Handle sparse matrices
    if sparse.issparse(expected) or sparse.issparse(actual):
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Array shape mismatch: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"Array values mismatch, max difference: {max_diff}"
        return True, "Arrays match"
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol)
            if not passed:
                return False, f"Mismatch at index {i}: {msg}"
        return True, "Lists/tuples match"
    
    # Handle dicts
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Dict keys mismatch: expected {set(expected.keys())}, got {set(actual.keys())}"
        for key in expected:
            passed, msg = custom_recursive_check(expected[key], actual[key], rtol, atol)
            if not passed:
                return False, f"Mismatch at key '{key}': {msg}"
        return True, "Dicts match"
    
    # Handle scalars
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"Scalar mismatch: expected {expected}, got {actual}"
        return True, "Scalars match"
    
    # Fallback to equality check
    try:
        if expected == actual:
            return True, "Values equal"
        else:
            return False, f"Value mismatch: expected {expected}, got {actual}"
    except Exception as e:
        return False, f"Comparison failed with exception: {e}"


def main():
    # Data paths provided
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_build_diffusion_matrix.pkl']
    
    # Identify outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_diffusion_matrix.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_diffusion_matrix.pkl)")
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
    
    try:
        agent_operator = build_diffusion_matrix(*outer_args, **outer_kwargs)
        print("Successfully called build_diffusion_matrix with outer args/kwargs")
    except Exception as e:
        print(f"ERROR: Failed to call build_diffusion_matrix: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and execute
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Using factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner args/kwargs")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = custom_recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: custom_recursive_check failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}: {msg}")
                sys.exit(1)
            else:
                print(f"Verification passed for inner data: {inner_path}")
    else:
        # Scenario A: Simple function pattern
        print("Using simple function pattern (no inner data)")
        
        result = agent_operator
        expected = outer_data.get('output')
        
        # Compare results using custom check for sparse matrices
        try:
            passed, msg = custom_recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: custom_recursive_check failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()