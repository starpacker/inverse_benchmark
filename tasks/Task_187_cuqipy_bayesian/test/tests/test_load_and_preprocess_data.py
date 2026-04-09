import sys
import os
import dill
import traceback
import numpy as np

# Add the parent directory to path if needed
sys.path.insert(0, '/data/yjh/cuqipy_bayesian_sandbox_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def compare_linear_models(expected_model, actual_model):
    """
    Compare two CUQI LinearModel objects by checking their properties
    and behavior rather than identity.
    """
    try:
        # Check if both have the same type
        if type(expected_model).__name__ != type(actual_model).__name__:
            return False, f"Model types differ: {type(expected_model).__name__} vs {type(actual_model).__name__}"
        
        # Check domain and range geometry
        if hasattr(expected_model, 'domain_geometry') and hasattr(actual_model, 'domain_geometry'):
            exp_domain = expected_model.domain_geometry
            act_domain = actual_model.domain_geometry
            if hasattr(exp_domain, 'par_dim') and hasattr(act_domain, 'par_dim'):
                if exp_domain.par_dim != act_domain.par_dim:
                    return False, f"Domain dimension mismatch: {exp_domain.par_dim} vs {act_domain.par_dim}"
        
        if hasattr(expected_model, 'range_geometry') and hasattr(actual_model, 'range_geometry'):
            exp_range = expected_model.range_geometry
            act_range = actual_model.range_geometry
            if hasattr(exp_range, 'par_dim') and hasattr(act_range, 'par_dim'):
                if exp_range.par_dim != act_range.par_dim:
                    return False, f"Range dimension mismatch: {exp_range.par_dim} vs {act_range.par_dim}"
        
        # Test functional equivalence with a sample input
        if hasattr(expected_model, 'domain_dim'):
            dim = expected_model.domain_dim
            np.random.seed(123)
            test_input = np.random.randn(dim)
            
            exp_output = expected_model @ test_input
            act_output = actual_model @ test_input
            
            if not np.allclose(np.asarray(exp_output), np.asarray(act_output), rtol=1e-5, atol=1e-8):
                return False, "Model outputs differ for same input"
        
        return True, "Models are functionally equivalent"
    except Exception as e:
        # If we can't properly compare, check string representation
        exp_str = str(expected_model)
        act_str = str(actual_model)
        if exp_str == act_str:
            return True, "Models have same string representation"
        return False, f"Could not compare models: {e}"

def custom_recursive_check(expected, actual, path="output"):
    """
    Custom recursive check that handles CUQI LinearModel comparison.
    """
    # Handle None
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"None mismatch at {path}: expected {expected}, got {actual}"
    
    # Handle dict
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Dict keys mismatch at {path}: expected {set(expected.keys())}, got {set(actual.keys())}"
        
        for key in expected.keys():
            # Special handling for forward_model
            if key == 'forward_model':
                passed, msg = compare_linear_models(expected[key], actual[key])
                if not passed:
                    return False, f"Forward model mismatch at {path}['{key}']: {msg}"
            else:
                passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']")
                if not passed:
                    return passed, msg
        return True, ""
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8):
            return False, f"Array values mismatch at {path}"
        return True, ""
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return passed, msg
        return True, ""
    
    # Handle primitive types
    if isinstance(expected, (int, float, str, bool)):
        if expected != actual:
            return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    # For other types, try standard comparison or string comparison
    try:
        if expected == actual:
            return True, ""
    except:
        pass
    
    # Check string representation as fallback
    if str(expected) == str(actual):
        return True, ""
    
    return False, f"Value mismatch at {path}: expected {expected}, got {actual}"

def main():
    data_paths = ['/data/yjh/cuqipy_bayesian_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    if outer_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Phase 1: Execute the function
    try:
        np.random.seed(42)  # Match the seed used in data generation
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario
    if inner_paths:
        print("\n=== Scenario B: Factory Pattern ===")
        # Load inner data and execute operator
        try:
            with open(inner_paths[0], 'rb') as f:
                inner_data = dill.load(f)
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data['output']
            
            # Execute the operator
            actual_result = result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR in Scenario B: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n=== Scenario A: Simple Function ===")
        expected = outer_data['output']
        actual_result = result
    
    # Verification
    print("\n=== Verification ===")
    print(f"Expected type: {type(expected)}")
    print(f"Result type: {type(actual_result)}")
    
    # Use custom recursive check that handles LinearModel
    passed, msg = custom_recursive_check(expected, actual_result)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()