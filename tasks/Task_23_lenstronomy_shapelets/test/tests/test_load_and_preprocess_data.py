import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def extract_comparable_data(result):
    """
    Extract comparable data from the result dictionary.
    Handles complex objects like ImageData, PSF, LensModel by extracting their key attributes.
    """
    if not isinstance(result, dict):
        return result
    
    comparable = {}
    
    for key, value in result.items():
        if key == 'data_class':
            # Extract key attributes from ImageData object
            try:
                comparable[key] = {
                    'data': getattr(value, '_data', None),
                    'background_rms': getattr(value, 'background_rms', None),
                    'C_D': getattr(value, 'C_D', None) if hasattr(value, 'C_D') else None,
                }
                # Try to get coordinates if available
                if hasattr(value, 'ra_at_xy_0'):
                    comparable[key]['ra_at_xy_0'] = value.ra_at_xy_0
                if hasattr(value, 'dec_at_xy_0'):
                    comparable[key]['dec_at_xy_0'] = value.dec_at_xy_0
            except Exception as e:
                comparable[key] = f"ImageData extraction error: {e}"
        
        elif key == 'psf_class':
            # Extract key attributes from PSF object
            try:
                comparable[key] = {
                    'psf_type': getattr(value, 'psf_type', None),
                    'fwhm': getattr(value, 'fwhm', None),
                }
                if hasattr(value, 'kernel_point_source'):
                    comparable[key]['kernel_point_source'] = value.kernel_point_source
            except Exception as e:
                comparable[key] = f"PSF extraction error: {e}"
        
        elif key == 'lens_model_class':
            # Extract key attributes from LensModel object
            try:
                comparable[key] = {
                    'lens_model_list': getattr(value, 'lens_model_list', None),
                }
            except Exception as e:
                comparable[key] = f"LensModel extraction error: {e}"
        
        elif isinstance(value, np.ndarray):
            comparable[key] = value
        
        elif isinstance(value, (int, float, str, list, dict, tuple, bool, type(None))):
            comparable[key] = value
        
        else:
            # For other objects, try to get a string representation or skip
            try:
                comparable[key] = str(type(value))
            except:
                comparable[key] = "Unknown type"
    
    return comparable


def compare_results(expected, actual):
    """
    Compare expected and actual results, handling complex objects.
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return recursive_check(expected, actual)
    
    # Compare only the keys that are easy to verify
    keys_to_compare = ['image_data', 'image_clean', 'numPix', 'deltaPix', 
                       'background_rms', 'exp_time', 'kwargs_numerics', 'kwargs_lens']
    
    for key in keys_to_compare:
        if key in expected and key in actual:
            passed, msg = recursive_check(expected[key], actual[key])
            if not passed:
                return False, f"Mismatch in key '{key}': {msg}"
        elif key in expected and key not in actual:
            return False, f"Key '{key}' missing in actual result"
    
    # For complex objects, just verify they exist and are of correct type
    if 'data_class' in expected and 'data_class' in actual:
        # Check that both have the same image data
        try:
            exp_data = expected['data_class']._data if hasattr(expected['data_class'], '_data') else None
            act_data = actual['data_class']._data if hasattr(actual['data_class'], '_data') else None
            if exp_data is not None and act_data is not None:
                passed, msg = recursive_check(exp_data, act_data)
                if not passed:
                    return False, f"data_class._data mismatch: {msg}"
        except Exception as e:
            pass  # Skip if extraction fails
    
    if 'psf_class' in expected and 'psf_class' in actual:
        try:
            if hasattr(expected['psf_class'], 'psf_type') and hasattr(actual['psf_class'], 'psf_type'):
                if expected['psf_class'].psf_type != actual['psf_class'].psf_type:
                    return False, "psf_class.psf_type mismatch"
        except Exception as e:
            pass
    
    if 'lens_model_class' in expected and 'lens_model_class' in actual:
        try:
            if hasattr(expected['lens_model_class'], 'lens_model_list') and hasattr(actual['lens_model_class'], 'lens_model_list'):
                if expected['lens_model_class'].lens_model_list != actual['lens_model_class'].lens_model_list:
                    return False, "lens_model_class.lens_model_list mismatch"
        except Exception as e:
            pass
    
    return True, "All comparable fields match"


def main():
    data_paths = ['/home/yjh/lenstronomy_shapelets_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Find outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data with outer data...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Function returned type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle inner data if present (Scenario B)
    if inner_paths:
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output', None)
                
                # Execute the operator with inner data
                if callable(result):
                    print("Executing returned operator with inner data...")
                    final_result = result(*inner_args, **inner_kwargs)
                else:
                    print("Result is not callable, using as-is")
                    final_result = result
                
                # Compare results
                print("Comparing results...")
                passed, msg = compare_results(expected_output, final_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function test
        print("Scenario A detected: Simple function test")
        print("Comparing results...")
        
        passed, msg = compare_results(expected_output, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()