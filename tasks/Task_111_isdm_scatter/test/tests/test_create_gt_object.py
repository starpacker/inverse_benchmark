import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/isdm_scatter_sandbox_sandbox/run_code')

from agent_create_gt_object import create_gt_object
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/isdm_scatter_sandbox_sandbox/run_code/std_data/standard_data_create_gt_object.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        print(f"Loaded outer data: func_name={outer_data.get('func_name')}, args types={[type(a).__name__ for a in outer_args]}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 1: Run the function
    try:
        agent_result = create_gt_object(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result).__name__}")
    except Exception as e:
        print(f"ERROR executing create_gt_object: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory pattern
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data['output']
                print(f"Loaded inner data: func_name={inner_data.get('func_name')}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not callable(agent_result):
                print("ERROR: agent_result is not callable but inner data exists (Scenario B expected).")
                sys.exit(1)
            
            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing agent_result: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED (inner): {msg}")
                    sys.exit(1)
                else:
                    print("Inner test passed.")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_data['output']
        result = agent_result
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()