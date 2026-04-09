import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def convert_to_comparable(obj):
    """
    Recursively convert objects to comparable types.
    Handles configobj.Section -> dict conversion and other special cases.
    """
    if obj is None:
        return None
    
    # Handle configobj.Section and similar dict-like objects
    if hasattr(obj, 'keys') and hasattr(obj, 'items') and not isinstance(obj, dict):
        # Convert to regular dict recursively
        return {k: convert_to_comparable(v) for k, v in obj.items()}
    
    if isinstance(obj, dict):
        return {k: convert_to_comparable(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        converted = [convert_to_comparable(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    
    if isinstance(obj, np.ndarray):
        return obj
    
    if isinstance(obj, (int, float, str, bool, np.integer, np.floating)):
        return obj
    
    # For other types, try to convert to a basic type if possible
    return obj


def deep_convert_dict(obj):
    """
    Deep conversion that handles nested configobj.Section objects.
    """
    if obj is None:
        return None
    
    # Check if it's a dict-like object (including configobj.Section)
    obj_type = type(obj).__name__
    if obj_type == 'Section' or (hasattr(obj, 'keys') and hasattr(obj, 'items')):
        result = {}
        for k, v in obj.items():
            result[k] = deep_convert_dict(v)
        return result
    
    if isinstance(obj, dict):
        return {k: deep_convert_dict(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        converted = [deep_convert_dict(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    
    return obj


def main():
    # Data paths provided
    data_paths = ['/home/yjh/BayHunter_standalone/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path:
        print(f"Found outer data file: {outer_path}")
    else:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    print(f"Found {len(inner_paths)} inner data file(s)")
    
    # Phase 1: Load outer data and execute function
    print("\n=== Phase 1: Loading outer data ===")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    print("\n=== Executing load_and_preprocess_data ===")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        if not callable(result):
            print("ERROR: Result is not callable for closure pattern")
            sys.exit(1)
        
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                # Execute the operator
                actual_result = result(*inner_args, **inner_kwargs)
                
                # Convert both for comparison
                expected_converted = deep_convert_dict(expected)
                actual_converted = deep_convert_dict(actual_result)
                
                passed, msg = recursive_check(expected_converted, actual_converted)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed")
                    
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n=== Scenario A: Simple Function ===")
        
        # Convert both expected and actual results to handle configobj.Section
        expected_converted = deep_convert_dict(outer_output)
        actual_converted = deep_convert_dict(result)
        
        try:
            passed, msg = recursive_check(expected_converted, actual_converted)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()