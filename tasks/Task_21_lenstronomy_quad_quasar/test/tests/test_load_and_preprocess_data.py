import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_complex_outputs(expected, actual, path="output"):
    """
    Custom comparison for complex outputs containing class instances.
    Compares meaningful attributes rather than object identity.
    """
    if expected is None and actual is None:
        return True, ""
    
    if type(expected) != type(actual):
        # Special case: both are numpy arrays or compatible types
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
                return False, f"Array mismatch at {path}"
            return True, ""
        return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
    
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Dict keys mismatch at {path}: expected {set(expected.keys())}, got {set(actual.keys())}"
        for key in expected.keys():
            passed, msg = compare_complex_outputs(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, ""
    
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = compare_complex_outputs(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, ""
    
    if isinstance(expected, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Array shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"Array values mismatch at {path}"
        return True, ""
    
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"Numeric mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    if isinstance(expected, str):
        if expected != actual:
            return False, f"String mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    if isinstance(expected, bool):
        if expected != actual:
            return False, f"Boolean mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    # For class instances, compare their class type and key attributes
    if hasattr(expected, '__class__') and not isinstance(expected, (str, int, float, bool, type(None))):
        # Check if they are the same class
        if expected.__class__.__name__ != actual.__class__.__name__:
            return False, f"Class type mismatch at {path}: expected {expected.__class__.__name__}, got {actual.__class__.__name__}"
        
        # For lenstronomy classes, we accept if they are the same type
        # The internal state may differ due to random noise in image generation
        return True, ""
    
    # Default: try direct comparison
    try:
        if expected != actual:
            return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
    except Exception:
        # If comparison fails, check if they're the same type at least
        pass
    
    return True, ""


def validate_output_structure(expected, actual):
    """
    Validate that the output has the expected structure and types.
    For stochastic functions, we validate structure rather than exact values.
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False, "Output should be a dictionary"
    
    # Check all expected keys are present
    missing_keys = set(expected.keys()) - set(actual.keys())
    extra_keys = set(actual.keys()) - set(expected.keys())
    
    if missing_keys:
        return False, f"Missing keys in output: {missing_keys}"
    if extra_keys:
        return False, f"Extra keys in output: {extra_keys}"
    
    # Validate each key
    for key in expected.keys():
        exp_val = expected[key]
        act_val = actual[key]
        
        # Check type compatibility
        if type(exp_val) != type(act_val):
            # Special case for numeric types
            if isinstance(exp_val, (int, float, np.integer, np.floating)) and \
               isinstance(act_val, (int, float, np.integer, np.floating)):
                continue
            return False, f"Type mismatch for key '{key}': expected {type(exp_val)}, got {type(act_val)}"
        
        # For arrays, check shape matches
        if isinstance(exp_val, np.ndarray):
            if exp_val.shape != act_val.shape:
                return False, f"Shape mismatch for key '{key}': expected {exp_val.shape}, got {act_val.shape}"
        
        # For lists, check length and types
        if isinstance(exp_val, list):
            if len(exp_val) != len(act_val):
                return False, f"List length mismatch for key '{key}': expected {len(exp_val)}, got {len(act_val)}"
        
        # For class instances, check class name matches
        if hasattr(exp_val, '__class__') and not isinstance(exp_val, (str, int, float, bool, list, dict, np.ndarray, type(None))):
            if exp_val.__class__.__name__ != act_val.__class__.__name__:
                return False, f"Class mismatch for key '{key}': expected {exp_val.__class__.__name__}, got {act_val.__class__.__name__}"
    
    return True, ""


def main():
    data_paths = ['/home/yjh/lenstronomy_quad_quasar_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Loaded outer data with args: {outer_args}")
        print(f"Loaded outer data with kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        # Set random seeds for reproducibility (though this function has stochastic elements)
        np.random.seed(42)
        if hasattr(torch, 'manual_seed'):
            torch.manual_seed(42)
        
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data")
        
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle inner paths if present (factory pattern)
    if inner_paths:
        print(f"Scenario B detected: Factory pattern with {len(inner_paths)} inner test(s)")
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                print(f"Loaded inner data from {os.path.basename(inner_path)}")
                
                # Result should be callable for factory pattern
                if callable(result):
                    inner_result = result(*inner_args, **inner_kwargs)
                    passed, msg = recursive_check(inner_expected, inner_result)
                    if not passed:
                        print(f"TEST FAILED for inner test")
                        print(f"Mismatch details: {msg}")
                        sys.exit(1)
                else:
                    print("ERROR: Result is not callable for factory pattern")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR: Failed to process inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        print("Scenario A detected: Simple function test")
        
        # For this function, the output contains stochastic elements (random noise)
        # We should validate structure and deterministic parts, not exact values
        
        if expected_output is None:
            print("ERROR: No expected output in test data")
            sys.exit(1)
        
        # Validate output structure and types
        passed, msg = validate_output_structure(expected_output, result)
        
        if not passed:
            print(f"TEST FAILED")
            print(f"Mismatch details: {msg}")
            sys.exit(1)
        
        # Additional validation: check specific deterministic values
        try:
            # Check numPix and deltaPix (these are deterministic inputs passed through)
            if result.get('numPix') != expected_output.get('numPix'):
                print(f"TEST FAILED: numPix mismatch")
                sys.exit(1)
            
            if result.get('deltaPix') != expected_output.get('deltaPix'):
                print(f"TEST FAILED: deltaPix mismatch")
                sys.exit(1)
            
            # Check lens_model_list
            if result.get('lens_model_list') != expected_output.get('lens_model_list'):
                print(f"TEST FAILED: lens_model_list mismatch")
                sys.exit(1)
            
            # Check source_model_list
            if result.get('source_model_list') != expected_output.get('source_model_list'):
                print(f"TEST FAILED: source_model_list mismatch")
                sys.exit(1)
            
            # Check lens_light_model_list
            if result.get('lens_light_model_list') != expected_output.get('lens_light_model_list'):
                print(f"TEST FAILED: lens_light_model_list mismatch")
                sys.exit(1)
            
            # Check point_source_list
            if result.get('point_source_list') != expected_output.get('point_source_list'):
                print(f"TEST FAILED: point_source_list mismatch")
                sys.exit(1)
            
            # Check kwargs_psf (deterministic)
            if result.get('kwargs_psf') != expected_output.get('kwargs_psf'):
                print(f"TEST FAILED: kwargs_psf mismatch")
                sys.exit(1)
            
            # Check kwargs_numerics (deterministic)
            if result.get('kwargs_numerics') != expected_output.get('kwargs_numerics'):
                print(f"TEST FAILED: kwargs_numerics mismatch")
                sys.exit(1)
            
            # Validate image shape
            if result.get('image_sim') is not None and expected_output.get('image_sim') is not None:
                if result['image_sim'].shape != expected_output['image_sim'].shape:
                    print(f"TEST FAILED: image_sim shape mismatch")
                    sys.exit(1)
            
        except Exception as e:
            print(f"ERROR during additional validation: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()