import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Import scipy.sparse for handling sparse matrix comparison
from scipy.sparse import issparse

def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices."""
    if not issparse(expected) or not issparse(actual):
        return False, "One of the matrices is not sparse"
    
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
    
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
    """Custom recursive check that handles sparse matrices and numpy random generators."""
    # Handle sparse matrices
    if issparse(expected) and issparse(actual):
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle numpy random generators - just check they're the same type
    if hasattr(expected, 'bit_generator') and hasattr(actual, 'bit_generator'):
        # Both are numpy random generators, consider them equal
        return True, "Random generators matched by type"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}: {set(expected.keys())} vs {set(actual.keys())}"
        
        for k in expected.keys():
            passed, msg = custom_recursive_check(expected[k], actual[k], rtol, atol, path=f"{path}['{k}']")
            if not passed:
                return False, msg
        return True, "All dictionary entries match"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: {expected.shape} vs {actual.shape}"
        if expected.dtype != actual.dtype:
            # Try to compare values anyway if dtypes are compatible
            pass
        if np.issubdtype(expected.dtype, np.floating) or np.issubdtype(expected.dtype, np.complexfloating):
            if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                return False, f"Array values differ at {path}"
        else:
            if not np.array_equal(expected, actual):
                return False, f"Array values differ at {path}"
        return True, "Arrays match"
    
    # Handle lists
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False, f"List length mismatch at {path}: {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Lists match"
    
    # Handle tuples
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False, f"Tuple length mismatch at {path}: {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Tuples match"
    
    # Handle scalars
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(actual, (int, float, np.integer, np.floating)):
            if np.isclose(expected, actual, rtol=rtol, atol=atol):
                return True, "Scalars match"
            else:
                return False, f"Scalar mismatch at {path}: {expected} vs {actual}"
    
    # Handle strings and other types
    try:
        if expected == actual:
            return True, "Values match"
        else:
            return False, f"Value mismatch at {path}: {expected} vs {actual}"
    except Exception as e:
        # If comparison fails, try string representation
        if str(expected) == str(actual):
            return True, "Values match (string comparison)"
        return False, f"Comparison failed at {path}: {str(e)}"


def main():
    data_paths = ['/data/yjh/NoisePy_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
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
    
    try:
        agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully called load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR: Failed to call load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
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
            expected = inner_data.get('output')
            
            if not callable(agent_operator):
                print("ERROR: agent_operator is not callable for Scenario B")
                sys.exit(1)
            
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
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        result = agent_operator
        expected = outer_data.get('output')
        
        try:
            passed, msg = custom_recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("Verification passed")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()