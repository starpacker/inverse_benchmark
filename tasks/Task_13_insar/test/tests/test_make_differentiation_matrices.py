#!/usr/bin/env python
"""
Unit Test script for make_differentiation_matrices function.
Tests the function that generates derivative operators as sparse matrices.
"""

import sys
import os
import dill
import numpy as np
import traceback

# Import scipy sparse for type checking and comparison
from scipy import sparse as sp

# Import the target function
from agent_make_differentiation_matrices import make_differentiation_matrices

# Import verification utility
from verification_utils import recursive_check


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Compare two scipy sparse matrices.
    
    Args:
        expected: Expected sparse matrix
        actual: Actual sparse matrix
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        tuple: (passed, message)
    """
    # Check if both are sparse matrices
    if not sp.issparse(expected):
        return False, f"Expected is not a sparse matrix, got {type(expected)}"
    if not sp.issparse(actual):
        return False, f"Actual is not a sparse matrix, got {type(actual)}"
    
    # Check shapes
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    
    # Convert to same format for comparison
    expected_csr = expected.tocsr()
    actual_csr = actual.tocsr()
    
    # Check dtype
    if expected_csr.dtype != actual_csr.dtype:
        # Allow dtype mismatch but warn
        pass
    
    # Compare the dense arrays (for small matrices) or use element-wise comparison
    # Convert to dense for comparison if matrices are not too large
    if expected_csr.shape[0] * expected_csr.shape[1] < 10_000_000:
        expected_dense = expected_csr.toarray()
        actual_dense = actual_csr.toarray()
        
        if not np.allclose(expected_dense, actual_dense, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(expected_dense - actual_dense))
            return False, f"Sparse matrix values differ. Max difference: {max_diff}"
    else:
        # For large matrices, compare non-zero elements
        diff = expected_csr - actual_csr
        if diff.nnz > 0:
            max_diff = np.max(np.abs(diff.data))
            if max_diff > atol + rtol * np.max(np.abs(expected_csr.data)):
                return False, f"Sparse matrix values differ. Max difference: {max_diff}"
    
    return True, "Sparse matrices match"


def compare_results(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """
    Custom comparison function that handles scipy sparse matrices.
    
    Args:
        expected: Expected result
        actual: Actual result
        rtol: Relative tolerance
        atol: Absolute tolerance
        path: Current path in the data structure (for error messages)
    
    Returns:
        tuple: (passed, message)
    """
    # Handle None
    if expected is None and actual is None:
        return True, "Both are None"
    if expected is None or actual is None:
        return False, f"At {path}: One is None, other is not"
    
    # Handle scipy sparse matrices
    if sp.issparse(expected) or sp.issparse(actual):
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle tuples and lists
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)):
            return False, f"At {path}: Type mismatch - expected {type(expected)}, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"At {path}: Length mismatch - expected {len(expected)}, got {len(actual)}"
        
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = compare_results(e, a, rtol, atol, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "All elements match"
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"At {path}: Type mismatch - expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            return False, f"At {path}: Key mismatch"
        
        for key in expected:
            passed, msg = compare_results(expected[key], actual[key], rtol, atol, f"{path}[{key}]")
            if not passed:
                return False, msg
        return True, "All dict items match"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"At {path}: Type mismatch - expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"At {path}: Shape mismatch - expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"At {path}: Array values differ. Max difference: {max_diff}"
        return True, "Arrays match"
    
    # Handle scalars and other types
    try:
        if isinstance(expected, (int, float, np.number)):
            if not np.isclose(expected, actual, rtol=rtol, atol=atol):
                return False, f"At {path}: Value mismatch - expected {expected}, got {actual}"
            return True, "Values match"
        
        # For other types, use equality
        if expected != actual:
            return False, f"At {path}: Value mismatch - expected {expected}, got {actual}"
        return True, "Values match"
    except Exception as e:
        return False, f"At {path}: Comparison error - {str(e)}"


def find_data_files(data_paths):
    """
    Categorize data files into outer and inner data paths.
    
    Args:
        data_paths: List of data file paths
    
    Returns:
        tuple: (outer_path, inner_paths)
    """
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_differentiation_matrices.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def load_data(file_path):
    """
    Load data from a pickle file using dill.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        dict: Loaded data
    """
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    return data


def main():
    """Main test function."""
    # Define data paths
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_make_differentiation_matrices.pkl']
    
    # Find outer and inner data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_make_differentiation_matrices.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute function
        print(f"Loading outer data from: {outer_path}")
        outer_data = load_data(outer_path)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute the function
        print("Executing make_differentiation_matrices...")
        result = make_differentiation_matrices(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
        
        # Phase 2: Determine scenario and verify
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
            
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                inner_data = load_data(inner_path)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # The result from Phase 1 should be callable
                if not callable(result):
                    print("ERROR: Result from make_differentiation_matrices is not callable")
                    sys.exit(1)
                
                # Execute the operator
                actual_result = result(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = compare_results(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                
                print(f"Inner test passed: {msg}")
        else:
            # Scenario A: Simple function
            print("Scenario A detected: No inner data files found")
            
            expected = outer_data.get('output')
            
            if expected is None:
                print("ERROR: No expected output found in outer data")
                sys.exit(1)
            
            # Compare results using custom comparison
            passed, msg = compare_results(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"Comparison result: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Failed during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()