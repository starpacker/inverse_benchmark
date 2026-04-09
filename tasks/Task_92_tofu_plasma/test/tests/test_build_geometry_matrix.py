import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_build_geometry_matrix import build_geometry_matrix
from verification_utils import recursive_check

# Import scipy.sparse for proper comparison
from scipy import sparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Compare two sparse matrices element-wise.
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
    
    # Check number of non-zero elements
    if expected_csr.nnz != actual_csr.nnz:
        return False, f"Number of non-zero elements mismatch: expected {expected_csr.nnz}, got {actual_csr.nnz}"
    
    # Compare data arrays
    if not np.allclose(expected_csr.data, actual_csr.data, rtol=rtol, atol=atol):
        return False, "Sparse matrix data values do not match"
    
    # Compare indices
    if not np.array_equal(expected_csr.indices, actual_csr.indices):
        return False, "Sparse matrix column indices do not match"
    
    # Compare indptr
    if not np.array_equal(expected_csr.indptr, actual_csr.indptr):
        return False, "Sparse matrix row pointers do not match"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """
    Custom recursive check that handles sparse matrices properly.
    """
    # Handle sparse matrices
    if sparse.issparse(expected) or sparse.issparse(actual):
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle tuples and lists
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)):
            return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "All elements match"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"Type mismatch at {path}: expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"Array values do not match at {path}"
        return True, "Arrays match"
    
    # Handle numeric types
    if isinstance(expected, (int, float, np.number)):
        if not isinstance(actual, (int, float, np.number)):
            return False, f"Type mismatch at {path}: expected numeric, got {type(actual)}"
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
        return True, "Values match"
    
    # Handle dicts
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch at {path}: expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}"
        for key in expected:
            passed, msg = custom_recursive_check(expected[key], actual[key], rtol, atol, f"{path}[{key}]")
            if not passed:
                return False, msg
        return True, "Dicts match"
    
    # Default comparison
    if expected != actual:
        return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
    
    return True, "Match"


def main():
    # Define data paths
    data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data_build_geometry_matrix.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_geometry_matrix.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_geometry_matrix.pkl)")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        print(f"  - Number of args: {len(outer_args)}")
        print(f"  - Number of kwargs: {len(outer_kwargs)}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario
    if len(inner_paths) > 0:
        print(f"Detected Scenario B: Factory/Closure pattern with {len(inner_paths)} inner data file(s)")
        
        # Phase 1: Create the operator
        try:
            agent_operator = build_geometry_matrix(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created agent operator")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print("ERROR: Agent operator is not callable")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {os.path.basename(inner_path)}")
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed agent operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = custom_recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        print("Detected Scenario A: Simple function")
        
        # Execute the function
        try:
            result = build_geometry_matrix(*outer_args, **outer_kwargs)
            print("Successfully executed build_geometry_matrix")
        except Exception as e:
            print(f"ERROR: Failed to execute build_geometry_matrix: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected_output = outer_data.get('output')
        
        # Verify results using custom check
        try:
            passed, msg = custom_recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()