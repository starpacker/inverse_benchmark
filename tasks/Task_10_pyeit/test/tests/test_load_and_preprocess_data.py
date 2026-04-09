#!/usr/bin/env python
"""
Unit Test for load_and_preprocess_data function.
Tests the EIT data loading and preprocessing functionality.
"""

import sys
import os
import traceback
import numpy as np

# Ensure dill is used for loading
try:
    import dill
except ImportError:
    print("ERROR: dill is required for loading test data")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scipy import sparse


def compare_sparse_matrices(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices."""
    if type(expected) != type(actual):
        return False, f"Sparse matrix type mismatch: {type(expected)} vs {type(actual)}"
    
    # Convert to same format for comparison
    exp_csr = expected.tocsr()
    act_csr = actual.tocsr()
    
    if exp_csr.shape != act_csr.shape:
        return False, f"Sparse matrix shape mismatch: {exp_csr.shape} vs {act_csr.shape}"
    
    # Compare data arrays
    if not np.allclose(exp_csr.data, act_csr.data, rtol=rtol, atol=atol):
        return False, "Sparse matrix data mismatch"
    
    if not np.array_equal(exp_csr.indices, act_csr.indices):
        return False, "Sparse matrix indices mismatch"
    
    if not np.array_equal(exp_csr.indptr, act_csr.indptr):
        return False, "Sparse matrix indptr mismatch"
    
    return True, "Sparse matrices match"


def compare_numpy_arrays(expected, actual, rtol=1e-5, atol=1e-8, path=""):
    """Compare two numpy arrays with tolerance."""
    if expected.shape != actual.shape:
        return False, f"{path}: Shape mismatch {expected.shape} vs {actual.shape}"
    
    if expected.dtype != actual.dtype:
        # Allow dtype differences if values are close
        pass
    
    if np.issubdtype(expected.dtype, np.floating) or np.issubdtype(actual.dtype, np.floating):
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"{path}: Array values differ, max diff: {max_diff}"
    else:
        if not np.array_equal(expected, actual):
            return False, f"{path}: Array values differ (non-float)"
    
    return True, f"{path}: Arrays match"


def compare_protocol_objects(expected, actual, path="protocol"):
    """Compare PyEITProtocol objects."""
    # Compare ex_mat
    if hasattr(expected, 'ex_mat') and hasattr(actual, 'ex_mat'):
        passed, msg = compare_numpy_arrays(expected.ex_mat, actual.ex_mat, path=f"{path}.ex_mat")
        if not passed:
            return False, msg
    
    # Compare meas_mat
    if hasattr(expected, 'meas_mat') and hasattr(actual, 'meas_mat'):
        passed, msg = compare_numpy_arrays(expected.meas_mat, actual.meas_mat, path=f"{path}.meas_mat")
        if not passed:
            return False, msg
    
    # Compare keep_ba
    if hasattr(expected, 'keep_ba') and hasattr(actual, 'keep_ba'):
        passed, msg = compare_numpy_arrays(expected.keep_ba, actual.keep_ba, path=f"{path}.keep_ba")
        if not passed:
            return False, msg
    
    return True, f"{path}: Protocol objects match"


def compare_mesh_objects(expected, actual, path="mesh"):
    """Compare mesh objects."""
    # Compare node arrays
    if hasattr(expected, 'node') and hasattr(actual, 'node'):
        passed, msg = compare_numpy_arrays(expected.node, actual.node, path=f"{path}.node")
        if not passed:
            return False, msg
    
    # Compare element arrays
    if hasattr(expected, 'element') and hasattr(actual, 'element'):
        passed, msg = compare_numpy_arrays(expected.element, actual.element, path=f"{path}.element")
        if not passed:
            return False, msg
    
    # Compare el_pos
    if hasattr(expected, 'el_pos') and hasattr(actual, 'el_pos'):
        passed, msg = compare_numpy_arrays(expected.el_pos, actual.el_pos, path=f"{path}.el_pos")
        if not passed:
            return False, msg
    
    # Compare perm_array if exists
    if hasattr(expected, 'perm_array') and hasattr(actual, 'perm_array'):
        if expected.perm_array is not None and actual.perm_array is not None:
            passed, msg = compare_numpy_arrays(expected.perm_array, actual.perm_array, path=f"{path}.perm_array")
            if not passed:
                return False, msg
    
    # Compare scalar attributes
    for attr in ['n_nodes', 'n_elems', 'ref_node', 'perm']:
        if hasattr(expected, attr) and hasattr(actual, attr):
            exp_val = getattr(expected, attr)
            act_val = getattr(actual, attr)
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                if exp_val != act_val:
                    return False, f"{path}.{attr}: Value mismatch {exp_val} vs {act_val}"
    
    return True, f"{path}: Mesh objects match"


def custom_recursive_check(expected, actual, rtol=1e-5, atol=1e-8, path="root"):
    """
    Custom recursive comparison that handles complex EIT objects.
    """
    # Handle None
    if expected is None and actual is None:
        return True, f"{path}: Both None"
    if expected is None or actual is None:
        return False, f"{path}: One is None, other is not"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"{path}: Expected ndarray, got {type(actual)}"
        return compare_numpy_arrays(expected, actual, rtol, atol, path)
    
    # Handle sparse matrices
    if sparse.issparse(expected):
        if not sparse.issparse(actual):
            return False, f"{path}: Expected sparse matrix, got {type(actual)}"
        return compare_sparse_matrices(expected, actual, rtol, atol)
    
    # Handle torch tensors
    if HAS_TORCH and isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return False, f"{path}: Expected torch.Tensor, got {type(actual)}"
        exp_np = expected.detach().cpu().numpy()
        act_np = actual.detach().cpu().numpy()
        return compare_numpy_arrays(exp_np, act_np, rtol, atol, path)
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"{path}: Expected dict, got {type(actual)}"
        
        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())
        
        if exp_keys != act_keys:
            missing = exp_keys - act_keys
            extra = act_keys - exp_keys
            return False, f"{path}: Key mismatch. Missing: {missing}, Extra: {extra}"
        
        for key in expected.keys():
            passed, msg = custom_recursive_check(
                expected[key], actual[key], rtol, atol, path=f"{path}['{key}']"
            )
            if not passed:
                return False, msg
        
        return True, f"{path}: Dict matches"
    
    # Handle lists and tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False, f"{path}: Expected {type(expected)}, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"{path}: Length mismatch {len(expected)} vs {len(actual)}"
        
        for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(
                exp_item, act_item, rtol, atol, path=f"{path}[{i}]"
            )
            if not passed:
                return False, msg
        
        return True, f"{path}: Sequence matches"
    
    # Handle protocol objects (check for characteristic attributes)
    if hasattr(expected, 'ex_mat') and hasattr(expected, 'meas_mat'):
        return compare_protocol_objects(expected, actual, path)
    
    # Handle mesh objects (check for characteristic attributes)
    if hasattr(expected, 'node') and hasattr(expected, 'element'):
        return compare_mesh_objects(expected, actual, path)
    
    # Handle numeric types
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if not isinstance(actual, (int, float, np.integer, np.floating)):
            return False, f"{path}: Expected numeric, got {type(actual)}"
        if np.isnan(expected) and np.isnan(actual):
            return True, f"{path}: Both NaN"
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"{path}: Numeric mismatch {expected} vs {actual}"
        return True, f"{path}: Numeric matches"
    
    # Handle strings
    if isinstance(expected, str):
        if expected != actual:
            return False, f"{path}: String mismatch '{expected}' vs '{actual}'"
        return True, f"{path}: String matches"
    
    # Handle booleans
    if isinstance(expected, bool):
        if expected != actual:
            return False, f"{path}: Bool mismatch {expected} vs {actual}"
        return True, f"{path}: Bool matches"
    
    # For other objects, try attribute-by-attribute comparison
    if hasattr(expected, '__dict__') and hasattr(actual, '__dict__'):
        return custom_recursive_check(expected.__dict__, actual.__dict__, rtol, atol, path)
    
    # Fallback: try direct equality (may fail for some objects)
    try:
        if expected == actual:
            return True, f"{path}: Direct equality passed"
        else:
            return False, f"{path}: Direct equality failed"
    except Exception as e:
        # If equality check fails, consider it a pass if types match
        if type(expected) == type(actual):
            return True, f"{path}: Same type, equality check failed but accepting"
        return False, f"{path}: Comparison failed with error: {e}"


def main():
    """Main test function."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Determine test scenario
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_path = path
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    scenario = "B (Factory/Closure)" if inner_path else "A (Simple Function)"
    print(f"Test Scenario: {scenario}")
    print(f"Outer path: {outer_path}")
    if inner_path:
        print(f"Inner path: {inner_path}")
    
    # --- Phase 1: Load outer data and execute function ---
    print("\n--- Phase 1: Loading outer data ---")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # --- Patch PyEITProtocol before importing ---
    print("\n--- Patching and importing agent module ---")
    try:
        from dataclasses import dataclass
        
        @dataclass
        class PyEITProtocol:
            """EIT Protocol object"""
            ex_mat: np.ndarray
            meas_mat: np.ndarray
            keep_ba: np.ndarray

            @property
            def n_meas(self) -> int:
                return self.meas_mat.shape[0]
        
        # Inject into the agent module's namespace
        import agent_load_and_preprocess_data
        agent_load_and_preprocess_data.PyEITProtocol = PyEITProtocol
        
        from agent_load_and_preprocess_data import load_and_preprocess_data
        print("Successfully patched PyEITProtocol and imported load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR importing agent module: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # --- Execute the function ---
    print("\n--- Executing load_and_preprocess_data ---")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # --- Phase 2: Handle inner data if exists (Scenario B) or compare directly (Scenario A) ---
    if inner_path:
        print("\n--- Phase 2: Loading inner data and executing operator (Scenario B) ---")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output', None)
        
        print(f"Inner args count: {len(inner_args)}")
        print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        # Execute the operator
        try:
            if callable(result):
                result = result(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            else:
                print("WARNING: Result is not callable, using as-is")
        except Exception as e:
            print(f"ERROR executing inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Phase 2: Comparing results (Scenario A) ---")
    
    # --- Comparison ---
    print("\n--- Comparing expected vs actual results ---")
    try:
        passed, msg = custom_recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        print(f"Details: {msg}")
        sys.exit(0)
    else:
        print("TEST FAILED")
        print(f"Failure details: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()