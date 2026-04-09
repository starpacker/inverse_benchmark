import sys
import os
import dill
import numpy as np
import traceback

# Add the path to find the agent module
sys.path.insert(0, '/home/yjh/oopao_zernike_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_data_dicts(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Custom comparison for the data dictionary returned by load_and_preprocess_data.
    Focuses on comparing numerical values and key properties rather than complex objects.
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False, f"Expected dict types, got {type(expected)} and {type(actual)}"
    
    # Keys that contain numpy arrays we should compare
    array_keys = ['zernike_inverse', 'zernike_basis_2d', 'phase_map', 'opd_map', 'coordinate_x', 'coordinate_y']
    
    # Scalar keys to compare
    scalar_keys = ['n_iterations']
    
    # Check all expected keys exist in actual
    for key in expected.keys():
        if key not in actual:
            return False, f"Missing key '{key}' in actual result"
    
    # Compare numpy arrays
    for key in array_keys:
        if key in expected:
            exp_val = expected[key]
            act_val = actual.get(key)
            
            if exp_val is None and act_val is None:
                continue
            if exp_val is None or act_val is None:
                return False, f"Array '{key}': one is None, other is not"
            
            if not isinstance(exp_val, np.ndarray) or not isinstance(act_val, np.ndarray):
                return False, f"Array '{key}': expected numpy arrays, got {type(exp_val)} and {type(act_val)}"
            
            if exp_val.shape != act_val.shape:
                return False, f"Array '{key}': shape mismatch {exp_val.shape} vs {act_val.shape}"
            
            if not np.allclose(exp_val, act_val, rtol=rtol, atol=atol, equal_nan=True):
                max_diff = np.max(np.abs(exp_val - act_val))
                return False, f"Array '{key}': values differ, max difference = {max_diff}"
    
    # Compare scalar values
    for key in scalar_keys:
        if key in expected:
            if expected[key] != actual.get(key):
                return False, f"Scalar '{key}': {expected[key]} != {actual.get(key)}"
    
    # For complex objects (telescope, source, atmosphere, zernike_object), 
    # verify they exist and have the right type
    object_keys = ['telescope', 'source', 'atmosphere', 'zernike_object']
    for key in object_keys:
        if key in expected:
            exp_obj = expected[key]
            act_obj = actual.get(key)
            
            if exp_obj is None and act_obj is None:
                continue
            if act_obj is None:
                return False, f"Object '{key}' is missing in actual result"
            
            # Check same class name
            exp_class = exp_obj.__class__.__name__
            act_class = act_obj.__class__.__name__
            if exp_class != act_class:
                return False, f"Object '{key}': class mismatch {exp_class} vs {act_class}"
            
            # For telescope, check key properties
            if key == 'telescope':
                if hasattr(exp_obj, 'resolution') and hasattr(act_obj, 'resolution'):
                    if exp_obj.resolution != act_obj.resolution:
                        return False, f"Telescope resolution mismatch"
                if hasattr(exp_obj, 'D') and hasattr(act_obj, 'D'):
                    if not np.isclose(exp_obj.D, act_obj.D):
                        return False, f"Telescope diameter mismatch"
    
    return True, "All comparisons passed"


def main():
    data_paths = ['/home/yjh/oopao_zernike_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"[INFO] Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"[INFO] Executing load_and_preprocess_data with args and kwargs")
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[INFO] Function executed successfully")
        
    except Exception as e:
        print(f"ERROR during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Detected Scenario B: Factory/Closure pattern with {len(inner_paths)} inner file(s)")
        
        # Verify the operator is callable
        if not callable(actual_result):
            print(f"ERROR: Expected callable operator, got {type(actual_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"[INFO] Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                print(f"[INFO] Executing operator with inner args and kwargs")
                inner_result = actual_result(*inner_args, **inner_kwargs)
                
                print("[INFO] Starting verification for inner result...")
                passed, msg = recursive_check(inner_expected, inner_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function pattern
        print("[INFO] Detected Scenario A: Simple function pattern")
        print("[INFO] Starting verification...")
        
        # Use custom comparison for the complex data dictionary
        passed, msg = compare_data_dicts(expected_output, actual_result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()