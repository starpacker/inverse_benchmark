import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_normalize_image import normalize_image
from verification_utils import recursive_check


def main():
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_normalize_image.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_normalize_image.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_normalize_image.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        result = normalize_image(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute normalize_image with outer args/kwargs")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        # The result should be a callable operator
        if not callable(result):
            print(f"ERROR: Expected callable operator from normalize_image, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data file: {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
            except Exception as e:
                print(f"ERROR: Failed to extract args/kwargs/output from inner data")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args/kwargs")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Failed during recursive_check")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        expected = outer_data.get('output')
        
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during recursive_check")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()