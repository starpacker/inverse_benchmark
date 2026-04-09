import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check

def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices."""
    from scipy import sparse
    
    if type(expected) != type(actual):
        return False, f"Type mismatch: expected {type(expected)}, got {type(actual)}"
    
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    
    # Convert to dense for comparison if small enough, otherwise compare components
    if expected.shape[0] * expected.shape[1] < 1e6:
        expected_dense = expected.toarray()
        actual_dense = actual.toarray()
        if not np.allclose(expected_dense, actual_dense, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(expected_dense - actual_dense))
            return False, f"Sparse matrix values differ, max diff: {max_diff}"
    else:
        # Compare sparse components
        if expected.nnz != actual.nnz:
            return False, f"NNZ mismatch: expected {expected.nnz}, got {actual.nnz}"
        
        exp_csr = expected.tocsr()
        act_csr = actual.tocsr()
        
        if not np.allclose(exp_csr.data, act_csr.data, rtol=rtol, atol=atol):
            return False, "Sparse matrix data arrays differ"
        if not np.array_equal(exp_csr.indices, act_csr.indices):
            return False, "Sparse matrix indices differ"
        if not np.array_equal(exp_csr.indptr, act_csr.indptr):
            return False, "Sparse matrix indptr differ"
    
    return True, "Sparse matrices match"

def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """Custom recursive check that handles sparse matrices."""
    from scipy import sparse
    
    # Handle sparse matrices
    if sparse.issparse(expected) or sparse.issparse(actual):
        if not (sparse.issparse(expected) and sparse.issparse(actual)):
            return False, f"At {path}: one is sparse, other is not"
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"At {path}: shape mismatch {expected.shape} vs {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"At {path}: arrays differ, max diff: {max_diff}"
        return True, "Arrays match"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"At {path}: key mismatch. Expected {set(expected.keys())}, got {set(actual.keys())}"
        for key in expected.keys():
            passed, msg = custom_recursive_check(expected[key], actual[key], rtol, atol, path=f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, "Dictionaries match"
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"At {path}: length mismatch {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Sequences match"
    
    # Handle scalars
    if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
        if not np.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            return False, f"At {path}: scalar mismatch {expected} vs {actual}"
        return True, "Scalars match"
    
    # Handle other types
    try:
        if expected == actual:
            return True, "Values match"
        else:
            return False, f"At {path}: value mismatch {expected} vs {actual}"
    except Exception as e:
        return False, f"At {path}: comparison error: {str(e)}"

def main():
    """Main test function."""
    data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    print("\n[Phase 1] Loading outer data...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'unknown')
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"  Function name: {func_name}")
        print(f"  Args count: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    print("\n[Phase 1] Executing load_and_preprocess_data with outer args...")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("  Execution completed successfully.")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine pattern and compare
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n[Phase 2] Detected Factory/Closure pattern")
        
        # Result from Phase 1 should be callable
        if not callable(result):
            print(f"ERROR: Expected callable from outer function, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]  # Use first inner path
        print(f"  Loading inner data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"  Inner args count: {len(inner_args)}")
            print(f"  Inner kwargs keys: {list(inner_kwargs.keys())}")
            
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Execute the operator
        print("  Executing agent operator with inner args...")
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("  Inner execution completed successfully.")
        except Exception as e:
            print(f"ERROR: Agent operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n[Phase 2] Detected Simple Function pattern (no inner paths)")
        expected = outer_output
        # result is already set from Phase 1
    
    # Compare results
    print("  Comparing results...")
    try:
        passed, msg = custom_recursive_check(expected, result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print(f"\nERROR: Comparison failed: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Comparison failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()