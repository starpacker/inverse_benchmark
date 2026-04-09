import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_obsdata(obs1, obs2):
    """Compare two ehtim Obsdata objects by their key attributes."""
    try:
        # Compare basic attributes
        if obs1.ra != obs2.ra:
            return False, f"ra mismatch: {obs1.ra} vs {obs2.ra}"
        if obs1.dec != obs2.dec:
            return False, f"dec mismatch: {obs1.dec} vs {obs2.dec}"
        if obs1.rf != obs2.rf:
            return False, f"rf mismatch: {obs1.rf} vs {obs2.rf}"
        
        # Compare data arrays
        if not np.allclose(obs1.data['time'], obs2.data['time'], rtol=1e-5, atol=1e-8):
            return False, "data['time'] mismatch"
        if not np.allclose(obs1.data['u'], obs2.data['u'], rtol=1e-5, atol=1e-8):
            return False, "data['u'] mismatch"
        if not np.allclose(obs1.data['v'], obs2.data['v'], rtol=1e-5, atol=1e-8):
            return False, "data['v'] mismatch"
        if not np.allclose(obs1.data['vis'], obs2.data['vis'], rtol=1e-5, atol=1e-8):
            return False, "data['vis'] mismatch"
        if not np.allclose(obs1.data['sigma'], obs2.data['sigma'], rtol=1e-5, atol=1e-8):
            return False, "data['sigma'] mismatch"
        
        return True, "Obsdata objects match"
    except Exception as e:
        return False, f"Error comparing Obsdata: {e}"


def compare_dict_values(expected, actual, path=""):
    """Custom comparison for dictionary values, handling special types."""
    if type(expected) != type(actual):
        # Special case: both might be observation data objects
        exp_type = type(expected).__name__
        act_type = type(actual).__name__
        if 'Obsdata' in exp_type and 'Obsdata' in act_type:
            return compare_obsdata(expected, actual)
        return False, f"Type mismatch at {path}: expected {exp_type}, got {act_type}"
    
    # Handle Obsdata objects
    if type(expected).__name__ == 'Obsdata':
        return compare_obsdata(expected, actual)
    
    # Handle torch tensors
    if isinstance(expected, torch.Tensor):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: {expected.shape} vs {actual.shape}"
        if expected.dtype != actual.dtype:
            return False, f"Dtype mismatch at {path}: {expected.dtype} vs {actual.dtype}"
        if not torch.allclose(expected.float(), actual.float(), rtol=1e-5, atol=1e-8):
            return False, f"Value mismatch at {path}"
        return True, ""
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: {expected.shape} vs {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"Value mismatch at {path}"
        return True, ""
    
    # Handle lists
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = compare_dict_values(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}: {set(expected.keys())} vs {set(actual.keys())}"
        for key in expected:
            passed, msg = compare_dict_values(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle scalars (int, float, str, etc.)
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
            if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8):
                return False, f"Value mismatch at {path}: {expected} vs {actual}"
        elif expected != actual:
            return False, f"Value mismatch at {path}: {expected} vs {actual}"
        return True, ""
    
    # Default comparison
    if expected != actual:
        return False, f"Value mismatch at {path}: {expected} vs {actual}"
    
    return True, ""


def main():
    data_paths = ['/home/yjh/dpi_task1_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner data files
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
    
    print(f"Function name: {outer_data.get('func_name', 'N/A')}")
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print(f"\nExecuting load_and_preprocess_data...")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine comparison strategy
    if inner_paths:
        # Scenario B: Factory pattern
        print(f"\nScenario B: Factory pattern with {len(inner_paths)} inner data file(s)")
        
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            # Result should be callable (operator)
            if not callable(result):
                print(f"ERROR: Expected result to be callable (operator), got {type(result)}")
                sys.exit(1)
            
            print("Executing operator with inner args...")
            try:
                actual = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            print("Comparing results...")
            passed, msg = compare_dict_values(expected, actual, "output")
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function comparison
        print("\nScenario A: Simple function comparison")
        expected = outer_data.get('output')
        
        print("Comparing results...")
        passed, msg = compare_dict_values(expected, result, "output")
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()