#!/usr/bin/env python
"""
Unit Test script for load_and_preprocess_data function.
Tests the function by loading serialized test data and comparing results.
"""

import sys
import os
import dill
import torch
import numpy as np
import traceback
from scipy import sparse as sp

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Compare two scipy sparse matrices.
    
    Returns
    -------
    passed : bool
    msg : str
    """
    if type(expected) != type(actual):
        return False, f"Sparse matrix type mismatch: {type(expected)} vs {type(actual)}"
    
    if expected.shape != actual.shape:
        return False, f"Sparse matrix shape mismatch: {expected.shape} vs {actual.shape}"
    
    # Convert to same format for comparison
    exp_csr = expected.tocsr()
    act_csr = actual.tocsr()
    
    # Compare data arrays
    if not np.allclose(exp_csr.data, act_csr.data, rtol=rtol, atol=atol):
        return False, "Sparse matrix data values do not match"
    
    # Compare indices
    if not np.array_equal(exp_csr.indices, act_csr.indices):
        return False, "Sparse matrix indices do not match"
    
    # Compare indptr
    if not np.array_equal(exp_csr.indptr, act_csr.indptr):
        return False, "Sparse matrix indptr do not match"
    
    return True, "Sparse matrices match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """
    Custom recursive check that handles scipy sparse matrices.
    """
    # Handle sparse matrices
    if sp.issparse(expected) or sp.issparse(actual):
        if not sp.issparse(expected):
            return False, f"{path}: Expected is not sparse but actual is"
        if not sp.issparse(actual):
            return False, f"{path}: Expected is sparse but actual is not"
        passed, msg = compare_sparse_matrices(expected, actual, rtol, atol)
        if not passed:
            return False, f"{path}: {msg}"
        return True, "OK"
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"{path}: Expected dict but got {type(actual)}"
        
        if set(expected.keys()) != set(actual.keys()):
            missing = set(expected.keys()) - set(actual.keys())
            extra = set(actual.keys()) - set(expected.keys())
            return False, f"{path}: Key mismatch. Missing: {missing}, Extra: {extra}"
        
        for key in expected.keys():
            passed, msg = custom_recursive_check(
                expected[key], actual[key], rtol, atol, path=f"{path}['{key}']"
            )
            if not passed:
                return False, msg
        return True, "OK"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"{path}: Expected ndarray but got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"{path}: Shape mismatch {expected.shape} vs {actual.shape}"
        if expected.dtype != actual.dtype:
            # Allow dtype mismatch but warn
            pass
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"{path}: Array values differ. Max diff: {max_diff}"
        return True, "OK"
    
    # Handle torch tensors
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return False, f"{path}: Expected Tensor but got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"{path}: Tensor shape mismatch {expected.shape} vs {actual.shape}"
        exp_np = expected.detach().cpu().numpy()
        act_np = actual.detach().cpu().numpy()
        if not np.allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.max(np.abs(exp_np - act_np))
            return False, f"{path}: Tensor values differ. Max diff: {max_diff}"
        return True, "OK"
    
    # Handle lists and tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)):
            return False, f"{path}: Type mismatch {type(expected)} vs {type(actual)}"
        if len(expected) != len(actual):
            return False, f"{path}: Length mismatch {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "OK"
    
    # Handle scalars and other types
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if not isinstance(actual, (int, float, np.integer, np.floating)):
            return False, f"{path}: Expected numeric but got {type(actual)}"
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"{path}: Value mismatch {expected} vs {actual}"
        return True, "OK"
    
    # Handle strings
    if isinstance(expected, str):
        if expected != actual:
            return False, f"{path}: String mismatch '{expected}' vs '{actual}'"
        return True, "OK"
    
    # Handle None
    if expected is None:
        if actual is not None:
            return False, f"{path}: Expected None but got {actual}"
        return True, "OK"
    
    # Fallback: direct comparison
    try:
        if expected != actual:
            return False, f"{path}: Value mismatch {expected} vs {actual}"
    except Exception as e:
        return False, f"{path}: Comparison failed with error: {e}"
    
    return True, "OK"


def main():
    """Main test function."""
    # Data paths provided
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data with outer args/kwargs...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and compare
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Factory/Closure pattern with {len(inner_paths)} inner data file(s)")
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator/closure
                if callable(result):
                    print("Executing returned operator with inner args/kwargs...")
                    actual_result = result(*inner_args, **inner_kwargs)
                else:
                    print("Result is not callable, using as-is")
                    actual_result = result
                
                # Compare
                print("Comparing results...")
                passed, msg = custom_recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {inner_path}")
                    
            except Exception as e:
                print(f"ERROR during inner test: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function test
        print("Scenario A detected: Simple function test")
        expected = outer_data.get('output')
        
        try:
            print("Comparing results...")
            passed, msg = custom_recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()