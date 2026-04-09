import sys
import os
import dill
import numpy as np
import traceback

# Add the path for imports
sys.path.insert(0, '/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code')

from agent_build_observation_operator import build_observation_operator
from verification_utils import recursive_check

# Import scipy sparse for comparison
from scipy import sparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices."""
    if not sparse.issparse(expected) or not sparse.issparse(actual):
        return False, "One or both objects are not sparse matrices"
    
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    
    # Convert to same format for comparison
    exp_csr = expected.tocsr()
    act_csr = actual.tocsr()
    
    # Compare the data arrays
    if not np.allclose(exp_csr.data, act_csr.data, rtol=rtol, atol=atol):
        return False, "Sparse matrix data arrays do not match"
    
    # Compare indices
    if not np.array_equal(exp_csr.indices, act_csr.indices):
        return False, "Sparse matrix indices do not match"
    
    # Compare indptr
    if not np.array_equal(exp_csr.indptr, act_csr.indptr):
        return False, "Sparse matrix indptr do not match"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8):
    """Custom recursive check that handles sparse matrices."""
    # Handle sparse matrices
    if sparse.issparse(expected) and sparse.issparse(actual):
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Array shape mismatch: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol):
            return False, "Array values do not match"
        return True, "Arrays match"
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol)
            if not passed:
                return False, f"Element {i}: {msg}"
        return True, "All elements match"
    
    # Handle dicts
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch: expected {set(expected.keys())}, got {set(actual.keys())}"
        for key in expected:
            passed, msg = custom_recursive_check(expected[key], actual[key], rtol, atol)
            if not passed:
                return False, f"Key '{key}': {msg}"
        return True, "Dicts match"
    
    # Handle scalars
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"Scalar mismatch: expected {expected}, got {actual}"
        return True, "Scalars match"
    
    # Handle None
    if expected is None and actual is None:
        return True, "Both are None"
    
    # Fallback: try direct comparison
    try:
        if expected == actual:
            return True, "Objects are equal"
        else:
            return False, f"Objects differ: expected {type(expected)}, got {type(actual)}"
    except Exception as e:
        return False, f"Comparison failed: {e}"


def main():
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_build_observation_operator.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_observation_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_observation_operator.pkl)")
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
        agent_operator = build_observation_operator(*outer_args, **outer_kwargs)
        print("Successfully called build_observation_operator")
    except Exception as e:
        print(f"ERROR: Failed to call build_observation_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and execute
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Using factory/closure scenario with {len(inner_paths)} inner data file(s)")
        
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
                print("Successfully executed agent_operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = custom_recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
                print(f"Verification passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: custom_recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Using simple function scenario (no inner data)")
        result = agent_operator
        expected = outer_data.get('output')
        
        try:
            passed, msg = custom_recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("Verification passed")
        except Exception as e:
            print(f"ERROR: custom_recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()