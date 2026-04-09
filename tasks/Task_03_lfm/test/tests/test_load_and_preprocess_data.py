import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Try to import scipy.sparse for sparse matrix handling
try:
    import scipy.sparse as sp
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


def custom_recursive_check(expected, actual, path="root", rtol=1e-5, atol=1e-8):
    """
    Recursively compare expected and actual values with support for various types.
    """
    # Handle None
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"At {path}: one is None, other is not (expected={expected}, actual={actual})"
    
    # Handle scipy sparse matrices
    if HAS_SCIPY_SPARSE and sp.issparse(expected):
        if not sp.issparse(actual):
            return False, f"At {path}: expected sparse matrix, got {type(actual)}"
        # Convert to dense for comparison or compare as arrays
        try:
            expected_dense = expected.toarray()
            actual_dense = actual.toarray()
            if expected_dense.shape != actual_dense.shape:
                return False, f"At {path}: sparse matrix shape mismatch {expected_dense.shape} vs {actual_dense.shape}"
            if np.allclose(expected_dense, actual_dense, rtol=rtol, atol=atol, equal_nan=True):
                return True, ""
            else:
                max_diff = np.max(np.abs(expected_dense - actual_dense))
                return False, f"At {path}: sparse matrix values differ, max diff={max_diff}"
        except Exception as e:
            # Try comparing sparse attributes
            try:
                if expected.shape != actual.shape:
                    return False, f"At {path}: sparse matrix shape mismatch"
                if expected.nnz != actual.nnz:
                    return False, f"At {path}: sparse matrix nnz mismatch"
                return True, ""  # Assume equal if shape and nnz match
            except Exception as e2:
                return False, f"At {path}: sparse matrix comparison failed: {e}, {e2}"
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"At {path}: expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            missing = set(expected.keys()) - set(actual.keys())
            extra = set(actual.keys()) - set(expected.keys())
            return False, f"At {path}: key mismatch. Missing: {missing}, Extra: {extra}"
        for key in expected.keys():
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']", rtol, atol)
            if not passed:
                return False, msg
        return True, ""
    
    # Handle lists and tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False, f"At {path}: expected list/tuple, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"At {path}: length mismatch {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]", rtol, atol)
            if not passed:
                return False, msg
        return True, ""
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"At {path}: expected numpy array, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"At {path}: shape mismatch {expected.shape} vs {actual.shape}"
        if expected.dtype != actual.dtype:
            # Allow dtype mismatch but try comparison
            pass
        try:
            # Handle object dtype arrays
            if expected.dtype == object or actual.dtype == object:
                if expected.shape != actual.shape:
                    return False, f"At {path}: object array shape mismatch"
                # Compare element by element
                for idx in np.ndindex(expected.shape):
                    passed, msg = custom_recursive_check(expected[idx], actual[idx], f"{path}[{idx}]", rtol, atol)
                    if not passed:
                        return False, msg
                return True, ""
            
            # Handle numeric arrays
            if np.issubdtype(expected.dtype, np.number) and np.issubdtype(actual.dtype, np.number):
                if np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                    return True, ""
                else:
                    max_diff = np.max(np.abs(expected - actual))
                    return False, f"At {path}: array values differ, max diff={max_diff}"
            
            # Handle boolean arrays
            if expected.dtype == bool and actual.dtype == bool:
                if np.array_equal(expected, actual):
                    return True, ""
                else:
                    return False, f"At {path}: boolean array mismatch"
            
            # Handle string arrays
            if np.issubdtype(expected.dtype, np.str_) or np.issubdtype(expected.dtype, np.bytes_):
                if np.array_equal(expected, actual):
                    return True, ""
                else:
                    return False, f"At {path}: string array mismatch"
            
            # Fallback: try array_equal
            if np.array_equal(expected, actual):
                return True, ""
            else:
                return False, f"At {path}: array mismatch"
                
        except Exception as e:
            return False, f"At {path}: array comparison failed: {e}"
    
    # Handle torch tensors
    try:
        import torch
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor):
                return False, f"At {path}: expected torch.Tensor, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"At {path}: tensor shape mismatch {expected.shape} vs {actual.shape}"
            expected_np = expected.detach().cpu().numpy()
            actual_np = actual.detach().cpu().numpy()
            if np.allclose(expected_np, actual_np, rtol=rtol, atol=atol, equal_nan=True):
                return True, ""
            else:
                max_diff = np.max(np.abs(expected_np - actual_np))
                return False, f"At {path}: tensor values differ, max diff={max_diff}"
    except ImportError:
        pass
    
    # Handle numeric scalars
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(actual, (int, float, np.integer, np.floating)):
            if np.isnan(expected) and np.isnan(actual):
                return True, ""
            if np.isclose(expected, actual, rtol=rtol, atol=atol):
                return True, ""
            else:
                return False, f"At {path}: numeric mismatch {expected} vs {actual}"
        return False, f"At {path}: expected numeric, got {type(actual)}"
    
    # Handle strings
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False, f"At {path}: expected str, got {type(actual)}"
        if expected == actual:
            return True, ""
        else:
            return False, f"At {path}: string mismatch '{expected}' vs '{actual}'"
    
    # Handle bytes
    if isinstance(expected, bytes):
        if not isinstance(actual, bytes):
            return False, f"At {path}: expected bytes, got {type(actual)}"
        if expected == actual:
            return True, ""
        else:
            return False, f"At {path}: bytes mismatch"
    
    # Handle complex numbers
    if isinstance(expected, (complex, np.complexfloating)):
        if isinstance(actual, (complex, np.complexfloating)):
            if np.isclose(expected, actual, rtol=rtol, atol=atol):
                return True, ""
            else:
                return False, f"At {path}: complex mismatch {expected} vs {actual}"
        return False, f"At {path}: expected complex, got {type(actual)}"
    
    # Handle objects with __dict__ (custom classes)
    if hasattr(expected, '__dict__') and hasattr(actual, '__dict__'):
        if type(expected).__name__ != type(actual).__name__:
            return False, f"At {path}: type mismatch {type(expected).__name__} vs {type(actual).__name__}"
        return custom_recursive_check(expected.__dict__, actual.__dict__, f"{path}.__dict__", rtol, atol)
    
    # Fallback: direct comparison
    try:
        if expected == actual:
            return True, ""
        else:
            return False, f"At {path}: value mismatch {expected} vs {actual}"
    except Exception as e:
        return False, f"At {path}: comparison failed with error: {e}"


def main():
    data_paths = ['/home/yjh/lfm_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer path and inner paths
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
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    print("Executing load_and_preprocess_data...")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario
    if inner_paths:
        print(f"Scenario B detected: Factory/Closure pattern with {len(inner_paths)} inner data file(s)")
        
        # Result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Executing operator with inner args...")
            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            print("Comparing results...")
            passed, msg = custom_recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print(f"Inner test passed for {os.path.basename(inner_path)}")
    else:
        print("Scenario A detected: Simple function")
        expected = outer_output
        
        print("Comparing results...")
        passed, msg = custom_recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()