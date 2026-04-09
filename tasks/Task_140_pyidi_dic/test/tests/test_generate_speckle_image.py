import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/pyidi_dic_sandbox_sandbox/run_code')

from agent_generate_speckle_image import generate_speckle_image
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data_generate_speckle_image.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_speckle_image.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_speckle_image.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    # Execute the function
    try:
        agent_result = generate_speckle_image(*outer_args, **outer_kwargs)
        print("Successfully executed generate_speckle_image")
    except Exception as e:
        print(f"ERROR: Failed to execute generate_speckle_image: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if we have inner paths (factory pattern) or simple function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result from outer call should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable from outer function but got non-callable")
            sys.exit(1)
        
        agent_operator = agent_result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        result = agent_result
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()