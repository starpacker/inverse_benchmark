import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__build_laplacian_2d import _build_laplacian_2d

# Import verification utility
from verification_utils import recursive_check

# Import scipy sparse for type checking
from scipy import sparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Compare two scipy sparse matrices.
    Returns (passed, message).
    """
    # Check if both are sparse matrices
    if not sparse.issparse(expected):
        return False, f"Expected is not a sparse matrix, got {type(expected)}"
    if not sparse.issparse(actual):
        return False, f"Actual is not a sparse matrix, got {type(actual)}"
    
    # Check shapes
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    
    # Check format (optional, convert if needed)
    expected_csr = expected.tocsr()
    actual_csr = actual.tocsr()
    
    # Check number of non-zero elements
    if expected_csr.nnz != actual_csr.nnz:
        return False, f"Number of non-zero elements mismatch: expected {expected_csr.nnz}, got {actual_csr.nnz}"
    
    # Compare data arrays
    if not np.allclose(expected_csr.data, actual_csr.data, rtol=rtol, atol=atol):
        return False, "Data values do not match"
    
    # Compare indices
    if not np.array_equal(expected_csr.indices, actual_csr.indices):
        return False, "Column indices do not match"
    
    # Compare indptr
    if not np.array_equal(expected_csr.indptr, actual_csr.indptr):
        return False, "Index pointers do not match"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Custom comparison that handles scipy sparse matrices.
    """
    # Handle sparse matrices
    if sparse.issparse(expected) or sparse.issparse(actual):
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
        return True, "All dict entries match"
    
    # Fallback to direct comparison
    try:
        if expected == actual:
            return True, "Values match"
        else:
            return False, f"Value mismatch: expected {expected}, got {actual}"
    except Exception as e:
        return False, f"Comparison error: {e}"


def main():
    # Define data paths
    data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data__build_laplacian_2d.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__build_laplacian_2d.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__build_laplacian_2d.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing _build_laplacian_2d with outer args...")
        agent_result = _build_laplacian_2d(*outer_args, **outer_kwargs)
        print(f"Agent result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute _build_laplacian_2d: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and compare
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Factory/Closure pattern")
        print(f"Found {len(inner_paths)} inner data file(s)")
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator with inner args
                if callable(agent_result):
                    print("Executing agent_result with inner args...")
                    result = agent_result(*inner_args, **inner_kwargs)
                else:
                    print("ERROR: Agent result is not callable for Scenario B")
                    sys.exit(1)
                
                # Compare results
                print("Comparing results...")
                passed, msg = custom_recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
                    
            except Exception as e:
                print(f"ERROR: Inner data processing failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function comparison
        print("\nScenario A detected: Simple function comparison")
        
        try:
            result = agent_result
            expected = expected_output
            
            print("Comparing results...")
            passed, msg = custom_recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"TEST PASSED: {msg}")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()