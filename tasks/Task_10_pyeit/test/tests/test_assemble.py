#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Test script for assemble function.
Tests the assembly of global stiffness matrix.
"""

import sys
import os
import dill
import numpy as np
import traceback
from scipy import sparse

# Import the target function
from agent_assemble import assemble

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
    if not sparse.issparse(expected):
        return False, f"Expected is not a sparse matrix, got {type(expected)}"
    if not sparse.issparse(actual):
        return False, f"Actual is not a sparse matrix, got {type(actual)}"
    
    # Check shapes
    if expected.shape != actual.shape:
        return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    
    # Convert to same format for comparison
    expected_csr = expected.tocsr()
    actual_csr = actual.tocsr()
    
    # Check number of non-zero elements
    if expected_csr.nnz != actual_csr.nnz:
        return False, f"Number of non-zero elements mismatch: expected {expected_csr.nnz}, got {actual_csr.nnz}"
    
    # Compare the dense arrays (for small matrices) or data arrays
    # Convert to dense for accurate comparison
    try:
        expected_dense = expected_csr.toarray()
        actual_dense = actual_csr.toarray()
        
        if not np.allclose(expected_dense, actual_dense, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(expected_dense - actual_dense))
            return False, f"Sparse matrix values differ. Max difference: {max_diff}"
        
        return True, "Sparse matrices match"
    except MemoryError:
        # For very large sparse matrices, compare data arrays directly
        # Sort indices first
        expected_csr.sort_indices()
        actual_csr.sort_indices()
        
        if not np.allclose(expected_csr.data, actual_csr.data, rtol=rtol, atol=atol):
            return False, "Sparse matrix data arrays differ"
        if not np.array_equal(expected_csr.indices, actual_csr.indices):
            return False, "Sparse matrix indices differ"
        if not np.array_equal(expected_csr.indptr, actual_csr.indptr):
            return False, "Sparse matrix indptr differ"
        
        return True, "Sparse matrices match (compared via internal arrays)"


def custom_recursive_check(expected, actual):
    """
    Custom recursive check that handles sparse matrices.
    
    Args:
        expected: Expected value
        actual: Actual value
    
    Returns:
        tuple: (passed, message)
    """
    # Handle sparse matrices specially
    if sparse.issparse(expected) or sparse.issparse(actual):
        return compare_sparse_matrices(expected, actual)
    
    # For non-sparse types, use the standard recursive_check
    return recursive_check(expected, actual)


def load_pickle_data(filepath):
    """
    Load data from a pickle file using dill.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        Loaded data dictionary
    """
    with open(filepath, 'rb') as f:
        data = dill.load(f)
    return data


def main():
    """Main test function."""
    
    # Define data paths
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_assemble.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_assemble.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_assemble.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute assemble
    try:
        outer_data = load_pickle_data(outer_path)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Determine scenario
    if len(inner_paths) > 0:
        print("Detected Scenario B: Factory/Closure Pattern")
        scenario = 'B'
    else:
        print("Detected Scenario A: Simple Function")
        scenario = 'A'
    
    # Execute the assemble function
    try:
        result = assemble(*outer_args, **outer_kwargs)
        print("Successfully executed assemble()")
    except Exception as e:
        print(f"ERROR: Failed to execute assemble(): {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle based on scenario
    if scenario == 'B':
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        print("Agent operator is callable")
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                inner_data = load_pickle_data(inner_path)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the agent operator
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator()")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator(): {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = custom_recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for {os.path.basename(inner_path)}: {msg}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple Function
        # The result is the output to compare
        expected = outer_data.get('output')
        
        # Verify results
        try:
            passed, msg = custom_recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print(f"Verification passed: {msg}")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()